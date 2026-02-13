# p = 0.2
python3 train_baseline_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.2 --num_classes 10 --exp_name baseline_sgd_mnt_cifar10n_0_2 --cuda 2 > train_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_2.txt  
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt_cifar10n_0_2.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_2.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.2 --num_classes 10 --exp_name asam_sgd_mnt_cifar10n_0_2 --adaptive True --cuda 2  > train_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_2.txt 
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_cifar10n_0_2.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_2.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.2 --num_classes 10 --exp_name sam_sgd_mnt_cifar10n_0_2 --adaptive False --rho 0.05 --cuda 2 > train_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_2.txt
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_cifar10n_0_2.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_2.txt  

# p = 0.4
python3 train_baseline_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.4 --num_classes 10 --exp_name baseline_sgd_mnt_cifar10n_0_4 --cuda 2 > train_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_4.txt  
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt_cifar10n_0_4.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_4.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.4 --num_classes 10 --exp_name asam_sgd_mnt_cifar10n_0_4 --adaptive True --cuda 2  > train_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_4.txt 
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_cifar10n_0_4.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_4.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.4 --num_classes 10 --exp_name sam_sgd_mnt_cifar10n_0_4 --adaptive False --rho 0.05 --cuda 2 > train_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_4.txt
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_cifar10n_0_4.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_4.txt  

# p = 0.6
python3 train_baseline_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.6 --num_classes 10 --exp_name baseline_sgd_mnt_cifar10n_0_6 --cuda 2 > train_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_6.txt  
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt_cifar10n_0_6.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_6.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.6 --num_classes 10 --exp_name asam_sgd_mnt_cifar10n_0_6 --adaptive True --cuda 2  > train_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_6.txt 
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_cifar10n_0_6.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_6.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.6 --num_classes 10 --exp_name sam_sgd_mnt_cifar10n_0_6 --adaptive False --rho 0.05 --cuda 2 > train_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_6.txt
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_cifar10n_0_6.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_6.txt  

# p = 0.8
python3 train_baseline_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name baseline_sgd_mnt_cifar10n_0_8 --cuda 2 > train_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_8.txt  
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_baseline_sgd_mnt_cifar10n_0_8.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name asam_sgd_mnt_cifar10n_0_8 --adaptive True --cuda 2  > train_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_8.txt 
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_asam_sgd_mnt_cifar10n_0_8.txt  

python3 train_sam_cifarn.py --arch resnet_50 --dataset cifar10n --noise_type uniform --noise_level 0.8 --num_classes 10 --exp_name sam_sgd_mnt_cifar10n_0_8 --adaptive False --rho 0.05 --cuda 2 > train_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_8.txt
python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_cifar10n_0_8.pth --eval_train True --cuda 2 > test_resnet_50_cifar10_sam_sgd_mnt_cifar10n_0_8.txt  


