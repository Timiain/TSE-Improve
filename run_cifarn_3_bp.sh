# p = 0.8
python3 train_baseline_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name baseline_sgd_mnt_cifar10n_0_8 --cuda 1 > train_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_8.txt  
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_8.txt  

python3 train_sam.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name asam_sgd_mnt_cifar10n_0_8 --adaptive True --cuda 1  > train_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_8.txt 
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_8.txt  

python3 train_sam.py --arch resnet_50 --dataset cifar10 --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name sam_sgd_mnt_cifar10n_0_8 --adaptive False --rho 0.05 --cuda 1 > train_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_8.txt
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_8.txt  

