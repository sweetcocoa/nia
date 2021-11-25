for MODEL in mobilenetv2 resnet18 resnet34
do
	CUDA_VISIBLE_DEVICES=2 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_baseline\' &
	sleep 10
done

for MODEL in mobilenetv2 resnet18 resnet34
do
	CUDA_VISIBLE_DEVICES=1 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_mel128\' pipe.n_mels=128 &
	sleep 10
done

for MODEL in mobilenetv2 resnet18 resnet34
do
	CUDA_VISIBLE_DEVICES=3 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_second_2\' pipe.target_audio_sample_length=32768 pipe.min_audio_sample_length=32768 &
	sleep 10
done