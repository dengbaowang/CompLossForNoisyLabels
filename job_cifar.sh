for (( i = 0; i < 3; i++ )); do
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.2 --noise_mode sym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.4 --noise_mode sym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.6 --noise_mode sym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.2 --noise_mode asym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.3 --noise_mode asym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.4 --noise_mode asym --rloss MAE --randidx $i
done

for (( i = 0; i < 3; i++ )); do
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.2 --noise_mode sym  --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.4 --noise_mode sym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.6 --noise_mode sym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.2 --noise_mode asym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.3 --noise_mode asym --rloss MAE --randidx $i
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.4 --noise_mode asym --rloss MAE --randidx $i
done