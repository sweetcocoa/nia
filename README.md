# NIA Dataset Trainer
---
## Scripts

### Docker Run
```bash
docker run \
        -it \
        --rm \
        --gpus all \
        -v /home/jonghochoi/docker/nia/data_midterm/:/data_midterm \
        -v /home/jonghochoi/docker/nia/data_segments/:/data_segments \
        -v /home/jonghochoi/docker/nia/lightning_logs/:/lightning_logs \
        -p 0.0.0.0:9015:9015 \
        nia:210914
```
```bash
python preprocess.py --input /data_midterm/ --output /data_segments/
python train.py
python inference_utils.py checkpoint=/path/to/ckpt
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
# Image file : confusion_matrix.jpg will be created in current folder.
```

- Code
```python
from omegaconf import OmegaConf
from inference_utils import Inference

config = OmegaConf.load("config.yaml")
inference = Inference("lightning_logs/version_0/checkpoints/model-epoch=0072-val_f1=0.872.ckpt", config)

# get confusion matrix (class-wise performance)
cm = inference.get_confusion_matrix_of_dataset(config.dataset.path, config, split='test', output_dir=".")
# >>> 

# inference one audio
inference.inference_audio("../data_segments/test/W_2_04/S-210909_W_204_D_001_0018_0.wav")
# >>> ("W_2_04", 0.9865)
```
