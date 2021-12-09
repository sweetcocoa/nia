# NIA Dataset Trainer
---
## Scripts

### Docker Run
```bash
docker run \
        -it \
        --rm \
        --gpus all \
        -v ${DATA_FOLDER}:/data_midterm \
        -v ${PREPROCESS_FOLDER}:/data_segments \
        -v ${LOG_FOLDER}:/lightning_logs \
        -p 0.0.0.0:9015:9015 \
        nia:211111
```
```bash
python preprocess.py --input_dir /data_midterm/ --output_dir /data_segments/ --minimum_sample 100 --sample_rate 16000
python train.py
python inference_utils.py --checkpoint /path/to/ckpt.pth
```

### Preprocess Audio (Extract Audio Segments, split data)

```bash
python preprocess --input /path/to/containing/mp3_and_json --output /path/to/segment_data/
```

- Dataset Folder Preview

```
segment_data/
    > train/
        > class_id_1/
            > class_audio_1.wav
            > class_audio_2.wav
        > class_id_2/
            > class_audio_1.wav
            > class_audio_2.wav
        ...
    > val
        > class_id_1/
            > class_audio_1.wav
            > class_audio_2.wav
        > class_id_2/
            > class_audio_1.wav
            > class_audio_2.wav
        ...
    > test
        > class_id_1/
            > class_audio_1.wav
            > class_audio_2.wav
        > class_id_2/
            > class_audio_1.wav
            > class_audio_2.wav
        ...
```

### Reproduce All the Results
```bash
bash runme_to_training.sh
bash runme_to_inference.sh
```

### Training

```bash
python train.py dataset.path=/path/to/segment_data/ training.batch_size=128
```

### Tensorboard
```bash
tensorboard --logdir lightning_logs/ --bind_all --port 9015
```

### Jupyter Lab
```bash
jupyter lab --allow-root --port 9015 --no-browser --ip 0.0.0.0
```

### Inference
- Script
```bash
python inference_utils.py checkpoint=lightning_logs/version_0/checkpoints/model-epoch\=0031-val_f1\=0.621.ckpt
# Image file : confusion_matrix.jpg will be created in version folder.
```