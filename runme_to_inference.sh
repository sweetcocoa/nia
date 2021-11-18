HP_POSTFIX='h8_f6_log'
LOGDIR='/lightning_logs/default'

for MODEL in vgg16 mobilenetv2 resnet18
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=2 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='h8_f6_log_mel128'

for MODEL in vgg16 mobilenetv2 resnet18
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=3 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done

HP_POSTFIX='h8_f6_linear'

for MODEL in vgg16 mobilenetv2 resnet18
do
	MODEL_CKPT_DIR=${LOGDIR}/${MODEL}_${HP_POSTFIX}/checkpoints
	CUDA_VISIBLE_DEVICES=3 python inference_utils.py --checkpoint ${MODEL_CKPT_DIR}/$(ls ${MODEL_CKPT_DIR} | grep model) &
	sleep 1
done