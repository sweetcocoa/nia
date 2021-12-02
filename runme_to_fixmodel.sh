
MODEL='resnet50'
MEL=40
BATCH_SIZE=48

for SEC in 1 2 4
do
    HP_POSTFIX=sec${SEC}_mel${MEL}
    CUDA_VISIBLE_DEVICES=0 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_${HP_POSTFIX}\' pipe.target_audio_sample_length=$(expr $SEC \* 16384) pipe.min_audio_sample_length=$(expr $SEC \* 16384) pipe.n_mels=$MEL training.batch_size=$BATCH_SIZE &
    sleep 3
done

MEL=64
for SEC in 1 2 4
do
    HP_POSTFIX=sec${SEC}_mel${MEL}
    CUDA_VISIBLE_DEVICES=2 python train.py model=\'${MODEL}\' training.version=\'${MODEL}_${HP_POSTFIX}\' pipe.target_audio_sample_length=$(expr $SEC \* 16384) pipe.min_audio_sample_length=$(expr $SEC \* 16384) pipe.n_mels=$MEL training.batch_size=$BATCH_SIZE &
    sleep 3
done