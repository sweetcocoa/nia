import torch
import torch.nn as nn

from audio_to_mel_pipe import AudioToMelPipe
from model import SimpleAudioClassificationModel


class Inference:
    def __init__(self, checkpoint_path):
        self.class_to_idx = {
            "A_1_01": 0,
            "A_2_01": 1,
            "A_2_02": 2,
            "A_2_03": 3,
            "A_2_04": 4,
            "H_2_02": 5,
            "N_1_01": 6,
            "N_2_01": 7,
            "N_2_03": 8,
            "N_2_04": 9,
            "V_2_02": 10,
            "V_2_04": 11,
        }
        self.idx_to_class = [
            k[0]
            for k in sorted(
                [(k, v) for k, v in self.class_to_idx.items()], key=lambda x: x[1]
            )
        ]
        self.pipe = AudioToMelPipe(is_validation=True)
        self.model = self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        def build_weight_only(experiment_state_dict):
            model_state_dict = dict()
            for k, v in experiment_state_dict["state_dict"].items():
                model_state_dict[k[len("model.") :]] = v
            return model_state_dict

        model = SimpleAudioClassificationModel(len(self.class_to_idx))
        experiment_state_dict = torch.load(checkpoint_path)
        model_state_dict = build_weight_only(experiment_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def inference_audio(self, audio_path):
        melspectrogram = self.pipe.load_audio(
            audio_path, target_frame_length=None
        ).unsqueeze(0)
        logits = self.model(melspectrogram)
        index = nn.functional.softmax(logits, dim=-1).argmax().item()
        predicted_label = self.idx_to_class[index]
        return predicted_label
