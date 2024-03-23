import torch
from Getdata import AudioDataset
from Log4p.core import *
import torch
from train import CNNclassifyModel

class DetiveDetector:
    def __init__(self, model_path, device):
        self.model = CNNclassifyModel(num_classes=60)
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(model_path))
        self.log = LogManager().GetLogger("DetiveDetector")

    def test(self, dataset):
        self.model.to(self.device)
        self.model.eval()

        mfccs, label = dataset[0]  # 获取数据集中的第一个样本
        mfccs = mfccs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(mfccs)  # 模型推理
            predicted_class = torch.argmax(output).item()
            self.log.info(f"Predicted class: {predicted_class}")

# 创建数据集对象
audio_files = ["kokomi.wav"]
labels = [0]
dataset = AudioDataset(audio_files=audio_files, labels=labels, sr=44100, duration=3, transform=None)

# 创建检测器对象并进行测试
dev = DetiveDetector("model/classifyFurinaCharacter.pth", "cpu")
dev.test(dataset)
