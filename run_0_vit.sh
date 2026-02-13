nohup python3 train_baseline.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --exp_name baseline_sgd_mnt --cuda 0 --learning_rate 0.03 > train_vit_b_4_cifar10_baseline_sgd_mnt.txt  &
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_baseline_sgd_mnt.txt  

nohup python3 train_sam.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --exp_name asam_sgd_mnt --adaptive True --cuda 2  --learning_rate 0.01 > train_vit_b_4_cifar10_asam_sgd_mnt.txt &
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_asam_sgd_mnt.txt  

nohup python3 train_sam.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 3 --learning_rate 0.03 > train_vit_b_4_cifar10_sam_sgd_mnt.txt &
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_sam_sgd_mnt.txt  

nohup python3 train_baseline.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --exp_name baseline_sgd_mnt --cuda 0 --learning_rate 0.03  > train_vit_b_4_cifar100_baseline_sgd_mnt.txt  &
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint baseline_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_baseline_sgd_mnt.txt  

nohup python3 train_sam.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --exp_name asam_sgd_mnt --adaptive True --cuda 0 --learning_rate 0.03  > train_vit_b_4_cifar100_asam_sgd_mnt.txt &
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint asam_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_asam_sgd_mnt.txt  

nohup python3 train_sam.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --exp_name sam_sgd_mnt --adaptive False --rho 0.05 --cuda 0 --learning_rate 0.03  > train_vit_b_4_cifar100_sam_sgd_mnt.txt &
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint sam_sgd_mnt.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_sam_sgd_mnt.txt  

#cifar10
#step 1
python3 interpolation.py --exp_name iter_sam_to_asam  --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start sam_sgd_mnt.pth --checkpoint_end asam_sgd_mnt.pth
python3 test_interpolation.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --exp_name iter_sam_to_asam
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA asam_sgd_mnt.pth --checkpoint_parentB sam_sgd_mnt.pth --checkpoint iter_sam_to_asam/model_alpha_0.5.pth --exp_name twin_step_1_A_prime --alpha 0.9 --cuda 0 > train_vit_b_4_cifar10_step_0_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA asam_sgd_mnt.pth --checkpoint_parentB sam_sgd_mnt.pth --checkpoint iter_sam_to_asam/model_alpha_0.5.pth --exp_name twin_step_1_B_prime --alpha 0.1 --cuda 0 > train_vit_b_4_cifar10_step_0_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_1_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_0_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_1_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_0_B.txt  
python3 interpolation.py --exp_name step1_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start twin_step_1_A_prime.pth --checkpoint_end twin_step_1_B_prime.pth
python3 test_interpolation.py --exp_name step1_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10

#step2
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_1_A_prime.pth --checkpoint_parentB twin_step_1_B_prime.pth --checkpoint step1_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_2_A_prime --alpha 0.9 --cuda 0 > train_vit_b_4_cifar10_step_1_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_1_A_prime.pth --checkpoint_parentB twin_step_1_B_prime.pth --checkpoint step1_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_2_B_prime --alpha 0.1 --cuda 0 > train_vit_b_4_cifar10_step_1_B.txt &
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_2_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_1_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_2_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_1_B.txt  

python3 interpolation.py --exp_name step2_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start twin_step_2_A_prime.pth --checkpoint_end twin_step_2_B_prime.pth
nohup python3 test_interpolation.py --exp_name step2_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 &

#step3
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_2_A_prime.pth --checkpoint_parentB twin_step_2_B_prime.pth --checkpoint step2_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_3_A_prime --alpha 0.9 --cuda 0 > train_vit_b_4_cifar10_step_2_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_2_A_prime.pth --checkpoint_parentB twin_step_2_B_prime.pth --checkpoint step2_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_3_B_prime --alpha 0.1 --cuda 0 > train_vit_b_4_cifar10_step_2_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_3_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_2_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_3_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_2_B.txt  

python3 interpolation.py --exp_name step3_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start twin_step_3_A_prime.pth --checkpoint_end twin_step_3_B_prime.pth
nohup python3 test_interpolation.py --exp_name step3_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 &

#step4
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_3_A_prime.pth --checkpoint_parentB twin_step_3_B_prime.pth --checkpoint step3_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_4_A_prime --alpha 0.9 --cuda 0 > train_vit_b_4_cifar10_step_3_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_3_A_prime.pth --checkpoint_parentB twin_step_3_B_prime.pth --checkpoint step3_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_4_B_prime --alpha 0.1 --cuda 0 > train_vit_b_4_cifar10_step_3_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_4_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_3_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_4_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_3_B.txt  

python3 interpolation.py --exp_name step4_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start twin_step_4_A_prime.pth --checkpoint_end twin_step_4_B_prime.pth
nohup python3 test_interpolation.py --exp_name step4_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 &

#step5
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_4_A_prime.pth --checkpoint_parentB twin_step_4_B_prime.pth --checkpoint step4_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_5_A_prime --alpha 0.9 --cuda 0 > train_vit_b_4_cifar10_step_4_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_parentA twin_step_4_A_prime.pth --checkpoint_parentB twin_step_4_B_prime.pth --checkpoint step4_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_5_B_prime --alpha 0.1 --cuda 0 > train_vit_b_4_cifar10_step_4_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_5_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_4_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint twin_step_5_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar10_step_4_B.txt  

python3 interpolation.py --exp_name step5_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 --checkpoint_start twin_step_5_A_prime.pth --checkpoint_end twin_step_5_B_prime.pth
nohup python3 test_interpolation.py --exp_name step5_A_prime_to_B_prime --arch vit_b_4 --dataset cifar10 --num_classes 10 &



#cifar100
#step 1
python3 interpolation.py --exp_name iter_sam_to_asam  --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start sam_sgd_mnt.pth --checkpoint_end asam_sgd_mnt.pth
python3 test_interpolation.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --exp_name iter_sam_to_asam
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA asam_sgd_mnt.pth --checkpoint_parentB sam_sgd_mnt.pth --checkpoint iter_sam_to_asam/model_alpha_0.5.pth --exp_name twin_step_1_A_prime --alpha 0.9 --cuda 2 > train_vit_b_4_cifar100_step_0_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA asam_sgd_mnt.pth --checkpoint_parentB sam_sgd_mnt.pth --checkpoint iter_sam_to_asam/model_alpha_0.5.pth --exp_name twin_step_1_B_prime --alpha 0.1 --cuda 2 > train_vit_b_4_cifar100_step_0_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_1_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_0_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_1_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_0_B.txt

python3 interpolation.py --exp_name step1_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start twin_step_1_A_prime.pth --checkpoint_end twin_step_1_B_prime.pth
python3 test_interpolation.py --exp_name step1_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --cuda 2

#step2
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_1_A_prime.pth --checkpoint_parentB twin_step_1_B_prime.pth --checkpoint step1_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_2_A_prime --alpha 0.9 --cuda 2 > train_vit_b_4_cifar100_step_1_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_1_A_prime.pth --checkpoint_parentB twin_step_1_B_prime.pth --checkpoint step1_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_2_B_prime --alpha 0.1 --cuda 2 > train_vit_b_4_cifar100_step_1_B.txt &
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_2_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_1_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_2_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_1_B.txt

python3 interpolation.py --exp_name step2_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start twin_step_2_A_prime.pth --checkpoint_end twin_step_2_B_prime.pth
nohup python3 test_interpolation.py --exp_name step2_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --cuda 2 &

#step3
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_2_A_prime.pth --checkpoint_parentB twin_step_2_B_prime.pth --checkpoint step2_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_3_A_prime --alpha 0.9 --cuda 2 > train_vit_b_4_cifar100_step_2_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_2_A_prime.pth --checkpoint_parentB twin_step_2_B_prime.pth --checkpoint step2_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_3_B_prime --alpha 0.1 --cuda 2 > train_vit_b_4_cifar100_step_2_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_3_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_2_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_3_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_2_B.txt

python3 interpolation.py --exp_name step3_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start twin_step_3_A_prime.pth --checkpoint_end twin_step_3_B_prime.pth
nohup python3 test_interpolation.py --exp_name step3_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --cuda 2 &

#step4
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_3_A_prime.pth --checkpoint_parentB twin_step_3_B_prime.pth --checkpoint step3_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_4_A_prime --alpha 0.9 --cuda 2 > train_vit_b_4_cifar100_step_3_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_3_A_prime.pth --checkpoint_parentB twin_step_3_B_prime.pth --checkpoint step3_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_4_B_prime --alpha 0.1 --cuda 2 > train_vit_b_4_cifar100_step_3_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_4_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_3_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_4_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_3_B.txt

python3 interpolation.py --exp_name step4_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start twin_step_4_A_prime.pth --checkpoint_end twin_step_4_B_prime.pth
nohup python3 test_interpolation.py --exp_name step4_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --cuda 2 &

#step5
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_4_A_prime.pth --checkpoint_parentB twin_step_4_B_prime.pth --checkpoint step4_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_5_A_prime --alpha 0.9 --cuda 2 > train_vit_b_4_cifar100_step_4_A.txt &
nohup python3 train_twin.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_parentA twin_step_4_A_prime.pth --checkpoint_parentB twin_step_4_B_prime.pth --checkpoint step4_A_prime_to_B_prime/model_alpha_0.5.pth --exp_name twin_step_5_B_prime --alpha 0.1 --cuda 2 > train_vit_b_4_cifar100_step_4_B.txt &

python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_5_A_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_4_A.txt  
python3 test.py --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint twin_step_5_B_prime.pth --eval_train True --cuda 0 > test_vit_b_4_cifar100_step_4_B.txt

python3 interpolation.py --exp_name step5_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --checkpoint_start twin_step_5_A_prime.pth --checkpoint_end twin_step_5_B_prime.pth
nohup python3 test_interpolation.py --exp_name step5_A_prime_to_B_prime --arch vit_b_4 --dataset cifar100 --num_classes 100 --cuda 2 &
