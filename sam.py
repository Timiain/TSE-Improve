from collections import deque

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


class CurvAdaptiveSAM(torch.optim.Optimizer):
    """CURV-ADAPT: curvature-aware SAM with projected gradients and adaptive schedule."""

    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        base_batch_size=128,
        min_batch_size=32,
        hessian_momentum=0.9,
        gp_rank=32,
        gp_window=20,
        gp_update_interval=10,
        gp_weight_temperature=5.0,
        curvature_floor=1e-6,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert base_batch_size > 0, "base_batch_size should be positive"
        assert min_batch_size > 0, "min_batch_size should be positive"
        assert 0.0 <= hessian_momentum < 1.0, "hessian_momentum should be in [0, 1)"
        assert gp_rank > 0, "gp_rank should be positive"
        assert gp_window > 0, "gp_window should be positive"
        assert gp_update_interval > 0, "gp_update_interval should be positive"

        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            base_batch_size=base_batch_size,
            min_batch_size=min_batch_size,
            hessian_momentum=hessian_momentum,
            gp_rank=gp_rank,
            gp_window=gp_window,
            gp_update_interval=gp_update_interval,
            gp_weight_temperature=gp_weight_temperature,
            curvature_floor=curvature_floor,
            **kwargs,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        for group in self.param_groups:
            if "base_lr" not in group:
                group["base_lr"] = group["lr"]

        self.gp_feature_history = deque(maxlen=gp_window)
        self.gp_loss_history = deque(maxlen=gp_window)
        self.curvature_error_var = 0.0
        self.global_step = 0

        self.last_d_plus = 0
        self.last_total_dim = 1
        self.last_eta_scale = 1.0
        self.last_sigma2 = 0.0
        self.suggested_batch_size = base_batch_size

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        projected_norm, d_plus, total_dim, feature_vec = self._project_gradients(update_curvature=True)
        self.last_d_plus = int(d_plus)
        self.last_total_dim = int(max(total_dim, 1))

        dim_ratio = d_plus / max(total_dim, 1)
        sigma2 = float(self.curvature_error_var)
        self.last_sigma2 = sigma2
        self.last_eta_scale = dim_ratio / (1.0 + sigma2)

        for group in self.param_groups:
            group["lr"] = group["base_lr"] * self.last_eta_scale
            scale = group["rho"] / (projected_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                perturb_grad = self.state[p]["projected_grad"]
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * perturb_grad * scale.to(p)
                p.add_(e_w)

            raw_batch = int(group["base_batch_size"] * dim_ratio)
            group["suggested_batch_size"] = max(group["min_batch_size"], raw_batch)
            self.suggested_batch_size = int(group["suggested_batch_size"])

        feature = feature_vec.detach().cpu()
        if feature.numel() > 0:
            self.gp_feature_history.append(feature)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        _, _, _, feature_vec = self._project_gradients(update_curvature=True)
        self.base_optimizer.step()
        self.global_step += 1

        feature = feature_vec.detach().cpu()
        if feature.numel() > 0:
            self.gp_feature_history.append(feature)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "CurvAdaptiveSAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        return loss

    @torch.no_grad()
    def update_surrogate(self, loss_value):
        self.gp_loss_history.append(float(loss_value))
        if self.global_step % self.param_groups[0]["gp_update_interval"] != 0:
            return
        if len(self.gp_feature_history) < 2 or len(self.gp_loss_history) < 2:
            return

        features = torch.stack(list(self.gp_feature_history))
        losses = torch.tensor(list(self.gp_loss_history), dtype=features.dtype)

        centered_loss = losses - losses.mean()
        temp = max(self.param_groups[0]["gp_weight_temperature"], 1e-6)
        weights = torch.softmax(-torch.abs(centered_loss) * temp, dim=0)

        mean_feature = torch.sum(features * weights[:, None], dim=0)
        diff = features - mean_feature
        weighted_var = torch.sum((diff.pow(2).sum(dim=1)) * weights) / max(features.shape[1], 1)
        self.curvature_error_var = float(weighted_var.item())

    @torch.no_grad()
    def _project_gradients(self, update_curvature=True):
        projected_norm_terms = []
        d_plus = 0
        total_dim = 0
        feature_chunks = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                flat_grad = grad.reshape(-1)
                if flat_grad.numel() == 0:
                    continue

                state = self.state[p]
                if "prev_grad" not in state:
                    state["prev_grad"] = torch.zeros_like(flat_grad)
                    state["h_diag_ema"] = torch.zeros_like(flat_grad)
                    state["feature_idx"] = self._init_feature_index(flat_grad.numel(), group["gp_rank"], flat_grad.device)
                    state["feature_sign"] = self._init_feature_sign(state["feature_idx"].numel(), flat_grad.device)

                grad_delta = flat_grad - state["prev_grad"]
                h_diag = state["h_diag_ema"] * group["hessian_momentum"] + (1.0 - group["hessian_momentum"]) * grad_delta.pow(2)
                h_diag = torch.clamp(h_diag, min=group["curvature_floor"])
                state["h_diag_ema"] = h_diag

                threshold = h_diag.mean()
                pos_mask = (h_diag >= threshold).to(flat_grad.dtype)
                projected = flat_grad * pos_mask

                residual = flat_grad - projected
                residual_norm_sq = torch.dot(residual, residual)
                if residual_norm_sq > 0:
                    correction_scale = torch.dot(projected, residual) / (residual_norm_sq + 1e-12)
                    projected = projected - correction_scale * residual

                projected_view = projected.view_as(grad)
                p.grad.copy_(projected_view)
                state["projected_grad"] = projected_view.clone()
                state["prev_grad"] = flat_grad.clone() if update_curvature else state["prev_grad"]

                d_plus += int(pos_mask.sum().item())
                total_dim += int(pos_mask.numel())
                projected_norm_terms.append(projected_view.norm(p=2))

                idx = state["feature_idx"]
                sign = state["feature_sign"]
                feature = flat_grad[idx] * sign
                feature_chunks.append(feature)

        if projected_norm_terms:
            projected_norm = torch.norm(torch.stack(projected_norm_terms), p=2)
        else:
            param = self.param_groups[0]["params"][0]
            projected_norm = torch.tensor(0.0, device=param.device)

        if feature_chunks:
            feature_vec = torch.cat(feature_chunks, dim=0)
        else:
            feature_vec = torch.tensor([], device=projected_norm.device)

        return projected_norm, d_plus, total_dim, feature_vec

    def _init_feature_index(self, n, rank, device):
        size = min(n, rank)
        if size == n:
            return torch.arange(n, device=device, dtype=torch.long)
        perm = torch.randperm(n, device=device)
        return perm[:size]

    def _init_feature_sign(self, n, device):
        return torch.randint(0, 2, (n,), device=device, dtype=torch.long).float().mul_(2.0).sub_(1.0)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class CAP(torch.optim.Optimizer):
    """Curvature-Aware Adaptive Perturbation with curvature-coupled radius and lr."""

    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        adaptive=False,
        c=0.1,
        alpha_min=1e-3,
        alpha_max=0.2,
        curvature_ema=0.05,
        lr_coupling=1.0,
        power_iter_steps=1,
        stability_lipschitz=10.0,
        k_sync=100,
        gamma_decay=0.95,
        c_decay=0.95,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 < alpha_min <= alpha_max, "alpha_min must be positive and <= alpha_max"
        assert 0.0 < curvature_ema <= 1.0, "curvature_ema must be in (0, 1]"
        assert power_iter_steps >= 1, "power_iter_steps must be >= 1"
        assert stability_lipschitz > 0.0, "stability_lipschitz must be positive"
        assert k_sync >= 1, "k_sync must be >= 1"

        defaults = dict(
            rho=rho,
            adaptive=adaptive,
            c=c,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            curvature_ema=curvature_ema,
            lr_coupling=lr_coupling,
            power_iter_steps=power_iter_steps,
            stability_lipschitz=stability_lipschitz,
            k_sync=k_sync,
            gamma_decay=gamma_decay,
            c_decay=c_decay,
            **kwargs,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        for group in self.param_groups:
            group.setdefault("base_lr", group["lr"])

        self.tau_ema = 0.0
        self.lambda_ema = 0.0
        self.last_tau = 0.0
        self.last_lambda = 0.0
        self.last_alpha = alpha_min
        self.last_lr = self.param_groups[0]["lr"]
        self.global_step = 0

    @torch.no_grad()
    def _restore_weights(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]

    def step(self, closure=None):
        assert closure is not None, "CAP requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        if not params:
            return None

        self.zero_grad()
        loss = closure()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        tau_t, lambda_t = self._estimate_curvature(params, grads)
        self.last_tau = float(tau_t.detach().item())
        self.last_lambda = float(lambda_t.detach().item())
        self._update_curvature_ema(tau_t, lambda_t)
        alpha_t = self._compute_alpha()
        lr_t = self._compute_lr(alpha_t)
        self.last_alpha = alpha_t
        self.last_lr = lr_t

        self._perturb_weights(grads, alpha_t)

        self.zero_grad()
        perturbed_loss = closure()
        perturbed_loss.backward()
        self._restore_weights()

        for group in self.param_groups:
            group["lr"] = lr_t

        self.base_optimizer.step()
        self.zero_grad()

        self.global_step += 1
        self._sync_hyperparams_if_needed()
        return loss.detach(), perturbed_loss.detach()

    def _estimate_curvature(self, params, grads):
        vector = self._rademacher_like(params)
        hvp_vector = self._hvp(grads, params, vector, retain_graph=True)
        tau_t = self._dot(vector, hvp_vector).abs()

        q = self._normalize_vector(vector)
        lambda_t = None
        for step in range(self.param_groups[0]["power_iter_steps"]):
            hvp_q = self._hvp(grads, params, q, retain_graph=step + 1 < self.param_groups[0]["power_iter_steps"])
            lambda_t = self._dot(q, hvp_q).abs()
            q = self._normalize_vector(hvp_q)

        if lambda_t is None:
            lambda_t = tau_t

        return tau_t, torch.clamp(lambda_t, min=1e-12)

    def _update_curvature_ema(self, tau_t, lambda_t):
        gamma = self.param_groups[0]["curvature_ema"]
        if self.global_step == 0:
            self.tau_ema = float(tau_t.detach().item())
            self.lambda_ema = float(lambda_t.detach().item())
            return

        self.tau_ema = (1.0 - gamma) * self.tau_ema + gamma * float(tau_t.detach().item())
        self.lambda_ema = (1.0 - gamma) * self.lambda_ema + gamma * float(lambda_t.detach().item())

    def _compute_alpha(self):
        group = self.param_groups[0]
        curvature_ratio = self.tau_ema / max(self.lambda_ema, 1e-12)
        alpha = group["c"] * curvature_ratio
        alpha = min(max(alpha, group["alpha_min"]), group["alpha_max"])

        max_alpha = 1.0 / max(2.0 * group["stability_lipschitz"] * group["base_lr"], 1e-12)
        alpha = min(alpha, max_alpha)
        return float(alpha)

    def _compute_lr(self, alpha_t):
        group = self.param_groups[0]
        lr_t = group["base_lr"] / (1.0 + group["lr_coupling"] * alpha_t)
        max_lr = 1.0 / max(2.0 * group["stability_lipschitz"] * max(alpha_t, 1e-12), 1e-12)
        return float(min(lr_t, max_lr))

    @torch.no_grad()
    def _perturb_weights(self, grads, alpha_t):
        grad_norm = self._grad_norm_from_sequence(grads)
        grad_index = 0
        for group in self.param_groups:
            scale = alpha_t / (grad_norm + 1e-12)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                grad = grads[grad_index]
                grad_index += 1
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * grad.detach() * scale.to(p)
                p.add_(e_w)

    def _sync_hyperparams_if_needed(self):
        group = self.param_groups[0]
        if self.global_step % group["k_sync"] != 0:
            return

        curvature_ratio = self.tau_ema / max(self.lambda_ema, 1e-12)
        if curvature_ratio < 0.25:
            group["c"] = min(group["alpha_max"], group["c"] / max(group["c_decay"], 1e-12))
        elif curvature_ratio > 1.0:
            group["c"] *= group["c_decay"]

        if self.lambda_ema > self.tau_ema:
            group["curvature_ema"] = min(0.5, group["curvature_ema"] / max(group["gamma_decay"], 1e-12))
        else:
            group["curvature_ema"] = max(1e-3, group["curvature_ema"] * group["gamma_decay"])

    def _hvp(self, grads, params, vector, retain_graph):
        hvp = torch.autograd.grad(
            grads,
            params,
            grad_outputs=vector,
            only_inputs=True,
            retain_graph=retain_graph,
            allow_unused=False,
        )
        return [h.detach() for h in hvp]

    def _rademacher_like(self, params):
        return [
            torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            for p in params
        ]

    def _normalize_vector(self, vector):
        norm = torch.sqrt(sum(torch.sum(v * v) for v in vector)).clamp_min(1e-12)
        return [v / norm for v in vector]

    def _dot(self, left, right):
        return sum(torch.sum(l * r) for l, r in zip(left, right))

    def _grad_norm_from_sequence(self, grads):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([g.detach().norm(p=2).to(shared_device) for g in grads]),
            p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
