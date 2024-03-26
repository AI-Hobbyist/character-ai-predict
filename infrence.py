import torch
from Getdata import AudioDataset
from Log4p.core import *
import torch,psutil
from character import character
from train import CNNclassifyModel
import json
from rich.progress import track
import argparse
from glob import glob
import os,gc

class DetiveDetector:
    def __init__(self, model_path, device):
        self.model = CNNclassifyModel(num_classes=60)
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.log = LogManager().GetLogger("DetiveDetector")

    def test(self, dataset):
        self.model.to(self.device)
        self.model.eval()

        mfccs, label = dataset[0]  # 获取数据集中的第一个样本
        mfccs = mfccs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(mfccs)  # 模型推理
            predicted_class = torch.argmax(output).item()
            if infoopt==True:
            	self.log.info(f"Predicted class: {predicted_class}")

        wr_dict = {audio_file:{"label": character[f"{predicted_class}"]},}

        if tftf == True:
            with open(json_opt, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_data.update(wr_dict)

            with open(json_opt, "w", encoding="utf-8") as f:
                json.dump(old_data, f, ensure_ascii=False, indent=2)
        else:
            with open(json_opt, "w", encoding="utf-8") as f:
                f.write(json.dumps(wr_dict, ensure_ascii=False, indent=2))

    

# 创建数据集对象
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scr', '--source', type=str, default = None, help='未整理数据集目录', required=True)
    parser.add_argument('-m', '--model', type=str, default = None, help='模型路径', required=True)
    parser.add_argument('-opt', '--output', type=str, default = "./opt.json", help='角色信息输出文件（json文件）（默认./opt.json）', required=False)
    parser.add_argument('-sr', '--sr', type=int, default = '44100', help='待处理数据集采样率（默认44100）', required=False)
    parser.add_argument('-info', '--info', type=str, default = False, help='是否在控制台输出推理结果（可选False/True，默认False）', required=False)
    args = parser.parse_args()

    audio_path=args.source
    model_path=args.model
    json_opt=args.output
    ausr=args.sr
    infoopt=args.info

    tftf=False

    files = glob(os.path.join(audio_path, "*.wav"))
    labels = [0]
    # 递归文件
    # 在循环外部创建 DetiveDetector 对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = DetiveDetector(model_path, device)

    # 循环中重复使用 dev
    for audio in track(files,description="[cyan]分类推理中..."):
        file = str(audio)
        audio_file = os.path.basename(file)
        atdl=[f"{audio_path}/{audio_file}"]
        dataset = AudioDataset(audio_files=atdl, labels=labels, sr=ausr, duration=3, transform=None)
        dev.test(dataset)
        tftf=True
        memory_useage = psutil.virtual_memory().used / (1024.0 ** 3)
        if memory_useage > 7:
            gc.collect()
    # 循环结束后手动释放资源
    del dev
    gc.collect()