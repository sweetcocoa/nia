# HP_POSTFIX='sec1'
# for MODEL in resnet18 resnet34 resnet50
# do
# 	CUDA_VISIBLE_DEVICES=2 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_${HP_POSTFIX}\' &
# 	sleep 10
# done

HP_POSTFIX='sec2'
for MODEL in resnet18
do
	CUDA_VISIBLE_DEVICES=3 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_${HP_POSTFIX}\' pipe.target_audio_sample_length=32768 pipe.min_audio_sample_length=32768
	# sleep 10
done

# HP_POSTFIX='sec4_bs48'
# for MODEL in resnet18 resnet34 resnet50
# do
# 	CUDA_VISIBLE_DEVICES=1 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_${HP_POSTFIX}\' pipe.target_audio_sample_length=65536 pipe.min_audio_sample_length=65536 training.batch_size=48 &
# 	sleep 10
# done

