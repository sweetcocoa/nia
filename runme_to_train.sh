
for MODEL in vgg16 mobilenetv2 resnet18
do
	CUDA_VISIBLE_DEVICES=2 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h8_f6_log\' pipe.use_log=True &
	sleep 10
done

for MODEL in vgg16 mobilenetv2 resnet18
do
	CUDA_VISIBLE_DEVICES=3 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h8_f6_linear\' pipe.use_log=False &
	sleep 10
done

for MODEL in vgg16 mobilenetv2 resnet18
do
	CUDA_VISIBLE_DEVICES=1 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h8_f6_log_mel128\' pipe.n_mels=128 pipe.use_log=True &
	sleep 10
done