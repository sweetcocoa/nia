HP_POSTFIX='sec1'
LOGDIR='/lightning_logs/dataset_1201/threshold_100/default/'

for MODEL in resnet18 resnet34 resnet50
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=2 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='sec2'

for MODEL in resnet18 resnet34 resnet50
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=3 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='sec4_bs48'
for MODEL in resnet18 resnet34 resnet50
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=1 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done



MODEL='resnet50'
MEL=40
BATCH_SIZE=48

for SEC in 1 2 4
do
    HP_POSTFIX=sec${SEC}_mel${MEL}
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=0 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

MEL=64
for SEC in 1 2 4
do
    HP_POSTFIX=sec${SEC}_mel${MEL}
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=2 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done