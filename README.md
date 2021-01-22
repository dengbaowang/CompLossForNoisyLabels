The implemention of our AAAI'21 paper (Learning from Noisy Labels with Complementary Loss Functions).

This implementation is based on PyTorch. You need:
1. 	Download CIFAR-10, CIFAR-100 and TinyImageNet datasets into './data/'.
2. 	Run the following Commands:
	```
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.4 --noise_mode sym --rloss MAE
	python Run_CIFAR.py --data_path ./data/cifar-100-python --dataset cifar100 --num_class 100 --r 0.3 --noise_mode asym --rloss MAE

	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.4 --noise_mode sym --rloss MAE
	python Run_CIFAR.py --data_path ./data/cifar-10-batches-py --dataset cifar10 --num_class 10 --r 0.3 --noise_mode asym --rloss MAE

	python Run_TinyImageNet.py --data_path ./data/tiny-imagenet-200 --dataset tiny --num_class 200 --r 0.5 --noise_mode sym --rloss MAE
	python Run_TinyImageNet.py --data_path ./data/tiny-imagenet-200 --dataset tiny --num_class 200 --r 0.3 --noise_mode asym --rloss MAE
  ```
	Or you can submit jobs using shell files like 'job_cifar.sh'. 

If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn.
