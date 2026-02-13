# python3 train_baseline.py --arch resnet_50 --dataset cifar10 --num_classes 10 --exp_name baseline_sgd_mnt --cuda 1 > train_resnet_50_cifar10_baseline_sgd_mnt.txt  
# python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_baseline_sgd_mnt.txt  

# python3 train_sam.py --arch resnet_50 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt --adaptive True --cuda 1  > train_resnet_50_cifar10_asam_sgd_mnt.txt 
# python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_asam_sgd_mnt.txt  

# python3 train_sam.py --arch resnet_50 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 1 > train_resnet_50_cifar10_sam_sgd_mnt.txt
# python3 test.py --arch resnet_50 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar10_sam_sgd_mnt.txt  

# python3 train_baseline.py --arch resnet_50 --dataset cifar100 --num_classes 100 --exp_name baseline_sgd_mnt --cuda 1 > train_resnet_50_cifar100_baseline_sgd_mnt.txt  
# python3 test.py --arch resnet_50 --dataset cifar100 --num_classes 100 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar100_baseline_sgd_mnt.txt  

# python3 train_sam.py --arch resnet_50 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt --adaptive True --cuda 1  > train_resnet_50_cifar100_asam_sgd_mnt.txt 
# python3 test.py --arch resnet_50 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar100_asam_sgd_mnt.txt  

# python3 train_sam.py --arch resnet_50 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 1 > train_resnet_50_cifar100_sam_sgd_mnt.txt
# python3 test.py --arch resnet_50 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 1 > test_resnet_50_cifar100_sam_sgd_mnt.txt  

#---------------------------------------SAM--cifar10
# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_3 --adaptive False --rho 0.001 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_n_3.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_n_3.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_2 --adaptive False --rho 0.01 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_n_2.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_n_2.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_1 --adaptive False --rho 0.1 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_n_1.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_n_1.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_0 --adaptive False --rho 1 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_0.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_0.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_1 --adaptive False --rho 10 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_1.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_1.txt  
# #---------------------------------------ASAM--cifar10
# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_3 --adaptive True --rho 0.001 --cuda 1 > train_resnet_18_cifar10_asam_sgd_mnt_10_n_3.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_asam_sgd_mnt_10_n_3.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_2 --adaptive True --rho 0.01 --cuda 1 > train_resnet_18_cifar10_asam_sgd_mnt_10_n_2.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_asam_sgd_mnt_10_n_2.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_1 --adaptive True --rho 0.1 --cuda 1 > train_resnet_18_cifar10_asam_sgd_mnt_10_n_1.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_asam_sgd_mnt_10_n_1.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_0 --adaptive True --rho 1 --cuda 1 > train_resnet_18_cifar10_asam_sgd_mnt_10_0.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_asam_sgd_mnt_10_0.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_1 --adaptive True --rho 10 --cuda 1 > train_resnet_18_cifar10_asam_sgd_mnt_10_1.txt
# python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_asam_sgd_mnt_10_1.txt  

# #---------------------------------------SAM--cifar100
# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_3 --adaptive False --rho 0.001 --cuda 1 > train_resnet_18_cifar100_sam_sgd_mnt_10_n_3.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_sam_sgd_mnt_10_n_3.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_2 --adaptive False --rho 0.01 --cuda 1 > train_resnet_18_cifar100_sam_sgd_mnt_10_n_2.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_sam_sgd_mnt_10_n_2.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_1 --adaptive False --rho 0.1 --cuda 1 > train_resnet_18_cifar100_sam_sgd_mnt_10_n_1.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_sam_sgd_mnt_10_n_1.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_0 --adaptive False --rho 1 --cuda 1 > train_resnet_18_cifar100_sam_sgd_mnt_10_0.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_sam_sgd_mnt_10_0.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_1 --adaptive False --rho 10 --cuda 1 > train_resnet_18_cifar100_sam_sgd_mnt_10_1.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_sam_sgd_mnt_10_1.txt  
# #---------------------------------------ASAM--cifar100
# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_3 --adaptive True --rho 0.001 --cuda 1 > train_resnet_18_cifar100_asam_sgd_mnt_10_n_3.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_asam_sgd_mnt_10_n_3.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_2 --adaptive True --rho 0.01 --cuda 1 > train_resnet_18_cifar100_asam_sgd_mnt_10_n_2.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_asam_sgd_mnt_10_n_2.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_1 --adaptive True --rho 0.1 --cuda 1 > train_resnet_18_cifar100_asam_sgd_mnt_10_n_1.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_asam_sgd_mnt_10_n_1.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_0 --adaptive True --rho 1 --cuda 1 > train_resnet_18_cifar100_asam_sgd_mnt_10_0.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_asam_sgd_mnt_10_0.txt  

# python3 train_sam.py --arch resnet_18 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_1 --adaptive True --rho 10 --cuda 1 > train_resnet_18_cifar100_asam_sgd_mnt_10_1.txt
# python3 test.py --arch resnet_18 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_resnet_18_cifar100_asam_sgd_mnt_10_1.txt  

#---------------------------------------SAM--cifar10
python3 train_sam.py --arch resnet_18 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_3 --adaptive False --rho 0.001 --cuda 1 > train_resnet_18_cifar10_sam_sgd_mnt_10_n_3.txt
python3 test.py --arch resnet_18 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_resnet_18_cifar10_sam_sgd_mnt_10_n_3.txt 

#---------------------------------------SAM--cifar10
python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_3 --adaptive False --rho 0.001 --cuda 1 > train_wrn28_2_cifar10_sam_sgd_mnt_10_n_3.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_sam_sgd_mnt_10_n_3.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_2 --adaptive False --rho 0.01 --cuda 1 > train_wrn28_2_cifar10_sam_sgd_mnt_10_n_2.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_sam_sgd_mnt_10_n_2.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_n_1 --adaptive False --rho 0.1 --cuda 1 > train_wrn28_2_cifar10_sam_sgd_mnt_10_n_1.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_sam_sgd_mnt_10_n_1.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_0 --adaptive False --rho 1 --cuda 1 > train_wrn28_2_cifar10_sam_sgd_mnt_10_0.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_sam_sgd_mnt_10_0.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt_10_1 --adaptive False --rho 10 --cuda 1 > train_wrn28_2_cifar10_sam_sgd_mnt_10_1.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_sam_sgd_mnt_10_1.txt  
#---------------------------------------ASAM--cifar10
python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_3 --adaptive True --rho 0.001 --cuda 1 > train_wrn28_2_cifar10_asam_sgd_mnt_10_n_3.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_asam_sgd_mnt_10_n_3.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_2 --adaptive True --rho 0.01 --cuda 1 > train_wrn28_2_cifar10_asam_sgd_mnt_10_n_2.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_asam_sgd_mnt_10_n_2.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_n_1 --adaptive True --rho 0.1 --cuda 1 > train_wrn28_2_cifar10_asam_sgd_mnt_10_n_1.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_asam_sgd_mnt_10_n_1.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_0 --adaptive True --rho 1 --cuda 1 > train_wrn28_2_cifar10_asam_sgd_mnt_10_0.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_asam_sgd_mnt_10_0.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt_10_1 --adaptive True --rho 10 --cuda 1 > train_wrn28_2_cifar10_asam_sgd_mnt_10_1.txt
python3 test.py --arch wrn28_2 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar10_asam_sgd_mnt_10_1.txt  

#---------------------------------------SAM--cifar100
python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_3 --adaptive False --rho 0.001 --cuda 1 > train_wrn28_2_cifar100_sam_sgd_mnt_10_n_3.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_sam_sgd_mnt_10_n_3.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_2 --adaptive False --rho 0.01 --cuda 1 > train_wrn28_2_cifar100_sam_sgd_mnt_10_n_2.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_sam_sgd_mnt_10_n_2.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_n_1 --adaptive False --rho 0.1 --cuda 1 > train_wrn28_2_cifar100_sam_sgd_mnt_10_n_1.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_sam_sgd_mnt_10_n_1.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_0 --adaptive False --rho 1 --cuda 1 > train_wrn28_2_cifar100_sam_sgd_mnt_10_0.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_sam_sgd_mnt_10_0.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt_10_1 --adaptive False --rho 10 --cuda 1 > train_wrn28_2_cifar100_sam_sgd_mnt_10_1.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_sam_sgd_mnt_10_1.txt  
#---------------------------------------ASAM--cifar100
python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_3 --adaptive True --rho 0.001 --cuda 1 > train_wrn28_2_cifar100_asam_sgd_mnt_10_n_3.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_3.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_asam_sgd_mnt_10_n_3.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_2 --adaptive True --rho 0.01 --cuda 1 > train_wrn28_2_cifar100_asam_sgd_mnt_10_n_2.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_2.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_asam_sgd_mnt_10_n_2.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_n_1 --adaptive True --rho 0.1 --cuda 1 > train_wrn28_2_cifar100_asam_sgd_mnt_10_n_1.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_n_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_asam_sgd_mnt_10_n_1.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_0 --adaptive True --rho 1 --cuda 1 > train_wrn28_2_cifar100_asam_sgd_mnt_10_0.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_0.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_asam_sgd_mnt_10_0.txt  

python3 train_sam.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt_10_1 --adaptive True --rho 10 --cuda 1 > train_wrn28_2_cifar100_asam_sgd_mnt_10_1.txt
python3 test.py --arch wrn28_2 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt_10_1.pth --eval_train True --cuda 1 > test_wrn28_2_cifar100_asam_sgd_mnt_10_1.txt  

