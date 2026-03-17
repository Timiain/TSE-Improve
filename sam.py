import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FGASAM(torch.optim.Optimizer):
    """Fourier-Gap Adaptive SAM with gradient-frequency projection."""

    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        alpha=0.2,
        beta=0.05,
        rho_min=0.0,
        rho_max=0.2,
        low_freq_ratio=0.25,
        proj_rank_ratio=0.1,
        validation_adjust_rate=0.05,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 <= low_freq_ratio < 1.0, "low_freq_ratio must be in [0, 1)"
        assert 0.0 <= proj_rank_ratio <= 1.0, "proj_rank_ratio must be in [0, 1]"
        assert rho_min <= rho_max, "rho_min should be <= rho_max"

        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            alpha=alpha,
            beta=beta,
            rho_min=rho_min,
            rho_max=rho_max,
            low_freq_ratio=low_freq_ratio,
            proj_rank_ratio=proj_rank_ratio,
            validation_adjust_rate=validation_adjust_rate,
            **kwargs,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.last_fourier_gap = 0.0
        self.last_rho = rho
        self._last_validation_loss = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        fourier_gap = self._fourier_gap()
        self.last_fourier_gap = float(fourier_gap.item()) if torch.is_tensor(fourier_gap) else float(fourier_gap)

        for group in self.param_groups:
            rho_t = group["alpha"] * fourier_gap + group["beta"]
            rho_t = torch.clamp(rho_t, min=group["rho_min"], max=group["rho_max"])
            self.last_rho = float(rho_t.item())
            scale = rho_t / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self._project_gradients()
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FGASAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def update_with_validation(self, validation_loss):
        """Light-weight schedule for alpha/beta and lr using validation feedback."""
        if self._last_validation_loss is None:
            self._last_validation_loss = float(validation_loss)
            return

        trend = float(validation_loss) - self._last_validation_loss
        self._last_validation_loss = float(validation_loss)

        for group in self.param_groups:
            rate = group["validation_adjust_rate"]
            if trend > 0:
                group["alpha"] *= 1.0 + rate
                group["beta"] *= 1.0 + rate
                group["lr"] *= 1.0 - 0.5 * rate
            else:
                group["alpha"] *= 1.0 - rate
                group["beta"] *= 1.0 - rate
                group["lr"] *= 1.0 + 0.5 * rate

            group["alpha"] = max(group["alpha"], 0.0)
            group["beta"] = max(group["beta"], 0.0)

    @torch.no_grad()
    def _project_gradients(self):
        for group in self.param_groups:
            rank_ratio = group["proj_rank_ratio"]
            if rank_ratio <= 0.0:
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_flat = p.grad.reshape(-1)
                if grad_flat.numel() < 4:
                    continue

                spectrum = torch.fft.rfft(grad_flat)
                energy = spectrum.abs().pow(2)
                k = max(1, int(rank_ratio * energy.numel()))
                if k >= energy.numel():
                    continue

                topk_idx = torch.topk(energy, k=k, largest=True).indices
                mask = torch.ones_like(spectrum)
                mask[topk_idx] = 0

                projected = torch.fft.irfft(spectrum * mask, n=grad_flat.numel())
                p.grad.copy_(projected.reshape_as(p.grad))

    def _fourier_gap(self):
        gaps = []
        for group in self.param_groups:
            low_ratio = group["low_freq_ratio"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad_flat = p.grad.reshape(-1)
                if grad_flat.numel() < 4:
                    continue

                spectrum = torch.fft.rfft(grad_flat)
                energy = spectrum.abs().pow(2)
                split = max(1, int(low_ratio * energy.numel()))
                low_energy = energy[:split].sum()
                high_energy = energy[split:].sum()
                total = low_energy + high_energy + 1e-12
                gaps.append((high_energy - low_energy) / total)

        if not gaps:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.stack(gaps).mean()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
