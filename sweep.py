import subprocess
import os
import GPUtil
import sys, time


def get_gpu():
    gpu = GPUtil.getAvailable(order="memory", limit=4, maxMemory=0.7, maxLoad=1.0)
    return gpu


dataset_root = "/data_segments/"
gpu = get_gpu()
print("gpu available : ", gpu)

for n_mels in [128, 40]:
    for sample_rate, hop_length, n_fft in [
        (16000, 256, 2048),
        (22050, 256, 2048),
        (44100, 512, 4096),
    ]:
        for duration in [4.0, 2.0, 1.0]:
            for model in ["resnet50", "resnet34", "resnet18"]:
                sample_length = int(sample_rate * duration * 1.024)
                dataset_path = os.path.join(
                    dataset_root, f"threshold_100_{sample_rate//1000}k"
                )
                training_version = (
                    f"{model}_sec{int(duration)}_sr{sample_rate//1000}k_nmel{n_mels}"
                )

                gpu = get_gpu()
                while not gpu:
                    sleep_sec = 60
                    print(f"no gpu available, sleep {sleep_sec}s...")
                    time.sleep(sleep_sec)
                    gpu = get_gpu()

                print("gpu available : ", gpu, f"Use GPU={gpu[0]}")
                args = f"model={model} training.version={training_version} dataset.path={dataset_path} pipe.n_fft={n_fft} pipe.hop_length={hop_length} pipe.sample_rate={sample_rate} pipe.target_audio_sample_length={sample_length} pipe.min_audio_sample_length={sample_length} pipe.n_mels={n_mels}"
                command = f"CUDA_VISIBLE_DEVICES={gpu[0]} /opt/conda/bin/python train.py {args}"
                print("command : {", command, "}")

                subprocess.Popen(
                    command,
                    shell=True,
                    # stdout=subprocess.DEVNULL,
                    # stderr=subprocess.DEVNULL,
                )
                time.sleep(30)

time.sleep(86400 * 10)
