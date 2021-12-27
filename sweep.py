import subprocess
import os
import GPUtil
import sys, time


def get_gpu():
    gpu = GPUtil.getAvailable(order="memory", limit=4, maxMemory=0.6, maxLoad=1.0)
    return gpu


dataset_root = "/data_segments/"
gpu = get_gpu()
print("gpu available : ", gpu)
cnt = 0

sample_rate, hop_length, n_fft = (22050, 256, 2048)

_lr = None
optimizer = "adam"
db_scale = False
scheduler = "multisteplr"
pretrained = True
n_mels = 128
duration = 4.0
augmentation = True
ws = True
label_smoothing = 0.1
masking = 0.25
model = "mobilenetv2"


def run():
    sample_length = int(sample_rate * duration * 1.024)
    dataset_name = f"211223_{sample_rate//1000}k"
    dataset_path = os.path.join(dataset_root, dataset_name)
    lr = _lr if _lr is not None else 1e-3

    training_version = f"_{model[-4:]}_ws_{str(ws)[0]}_nmel_{n_mels}_dur_{int(duration)}sec_aug_{str(augmentation)[0]}_masking{masking}_smooth{label_smoothing}_lr{_lr}"

    find_lr = True if _lr is None else False
    use_log = not db_scale

    gpu = get_gpu()
    while not gpu:
        sleep_sec = 60
        print(f"no gpu available, sleep {sleep_sec}s...")
        time.sleep(sleep_sec)
        gpu = get_gpu()

    """
    <--- Training --->
    """

    args = f"""model.name={model} \\
model.pretrained={pretrained} \\
training.optimizer={optimizer} \\
training.version={training_version} \\
training.logdir=/lightning_logs/{dataset_name} \\
training.auto_lr_find={find_lr} \\
training.lr_scheduler={scheduler} \\
training.learning_rate={lr} \\
training.weighted_sampling={ws} \\
training.label_smoothing={label_smoothing} \\
dataset.path={dataset_path} \\
pipe.n_fft={n_fft} \\
pipe.hop_length={hop_length} \\
pipe.sample_rate={sample_rate} \\
pipe.target_audio_sample_length={sample_length} \\
pipe.min_audio_sample_length={sample_length} \\
pipe.n_mels={n_mels} \\
pipe.use_log={use_log} \\
pipe.db_scale={db_scale} \\
pipe.augment.do={augmentation} \\
pipe.augment.freq_masking={masking} \\
pipe.augment.time_masking={masking}"""
    command = f"CUDA_VISIBLE_DEVICES={gpu[0]} /opt/conda/bin/python train.py {args}"
    print("command : {", command, "}")

    subprocess.Popen(
        command,
        shell=True,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    time.sleep(20)


for model in ["resnet34", "resnet50", "mobilenetv2"]:
    run()

model = "mobilenetv2"

for n_mels in [40, 80, 128]:
    run()

n_mels = 128

for duration in [1.0, 2.0, 4.0]:
    run()

duration = 4.0

for label_smoothing in [0.1, 0.0]:
    run()

label_smoothing = 0.1

for sample_rate, hop_length, n_fft in [(22050, 256, 2048), (16000, 256, 2048)]:
    run()

sample_rate, hop_length, n_fft = (22050, 256, 2048)

for _lr in [None, 1e-3, 5e-3, 5e-4]:
    run()

time.sleep(86400 * 10)
