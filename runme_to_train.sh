
for MODEL in vgg16 mobilenetv2 resnet18
do
	CUDA_VISIBLE_DEVICES=2 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h8_f6\' dataset.use_major_class=True &
	sleep 10
done

# for MODEL in vgg16 mobilenetv2 resnet18
# do
# 	CUDA_VISIBLE_DEVICES=1 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h7_f7\' pipe.hop_length=128 pipe.target_frame_length=128 &
# 	sleep 10
# done

for MODEL in vgg16 mobilenetv2 resnet18
do
	CUDA_VISIBLE_DEVICES=3 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h8_f6_minor\' dataset.use_major_class=False &
	sleep 10
done

# for MODEL in vgg16 mobilenetv2 resnet18
# do
# 	CUDA_VISIBLE_DEVICES=3 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_h7_f7_minor\' pipe.hop_length=128 pipe.target_frame_length=128 dataset.use_major_class=False &
# 	sleep 10
# done