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
        nia:211227
```

### Preprocess Audio (Extract Audio Segments, split data)

```bash
python preprocess --input /path/to/containing/mp3_and_json --output /path/to/segment_data/ --minimum_sample 100 --sample_rate 22050
```

- Preview of Preprocessed Dataset Folder 

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

### Reproduce One Results
---
```bash
python train.py dataset.path=/path/to/segment/folder 
```

### Reproduce All the Results
---
```bash
python sweep.py
```

- You can check result logs at /lightning_logs/

### Training

```bash
python train.py dataset.path=/path/to/segment_data/ training.batch_size=96
```

### Tensorboard
```bash
tensorboard --logdir lightning_logs/ --bind_all --port 9015
```