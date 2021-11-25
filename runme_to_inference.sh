HP_POSTFIX='baseline'
LOGDIR='/lightning_logs/threshold_100/default/'

for MODEL in mobilenetv2 resnet18 resnet34
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=2 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='mel128'

for MODEL in mobilenetv2 resnet18 resnet34
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=3 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='second_2'

for MODEL in mobilenetv2 resnet18 resnet34
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=1 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done
