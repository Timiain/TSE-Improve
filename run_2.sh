# python3 train_baseline.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name baseline_sgd_mnt --cuda 2 > train_wrn28_2_cifar10_baseline_sgd_mnt.txt  
# python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar10_baseline_sgd_mnt.txt  

# python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt --adaptive True --cuda 2  > train_wrn28_2_cifar10_asam_sgd_mnt.txt 
# python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar10_asam_sgd_mnt.txt  

# python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 2 > train_wrn28_2_cifar10_sam_sgd_mnt.txt
# python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar10_sam_sgd_mnt.txt  

# python3 train_baseline.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name baseline_sgd_mnt --cuda 2 > train_wrn28_2_cifar100_baseline_sgd_mnt.txt  
# python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar100_baseline_sgd_mnt.txt  

# python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt --adaptive True --cuda 2  > train_wrn28_2_cifar100_asam_sgd_mnt.txt 
# python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar100_asam_sgd_mnt.txt  

# python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 2 > train_wrn28_2_cifar100_sam_sgd_mnt.txt
# python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 2 > test_wrn28_2_cifar100_sam_sgd_mnt.txt  
