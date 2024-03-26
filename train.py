from Log4p.core import *
from config import Config
from Getdata import AudioDataset,AudioAugmentation
from GetPath import GetDataPath
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import global_exc_handler
from rich.progress import Progress
import torch,os
from Decorators import retry
import json

# 定义CNN游戏角色语音分类模型
class CNNclassifyModel(nn.Module):
    def __init__(self, num_classes, reg_lambda=0.01):
        super(CNNclassifyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
        self.reg_lambda = reg_lambda

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adapt_pool(x)
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def l2_regularization_loss(self):
        l2_reg = torch.tensor(0., device=self.fc2.weight.device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.reg_lambda * l2_reg

class Trainer:
    def __init__(self,model:nn.Module, data_loader:DataLoader, batch_size:int, learning_rate:float, num_epochs:int, device:str):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device(device=device)
        # 损失函数和优化器不要忘了加
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.log = LogManager().GetLogger("TrainThread")
        self.progress = Progress()
        self.task = self.progress.add_task("[cyan]训练中...", total=self.num_epochs)

    # 模型准确率
    def Accuracy(self, model:nn.Module, data_loader:DataLoader, device:torch.device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for mfccs, labels in data_loader:
                mfccs = mfccs.to(device)
                labels = labels.to(device)
                outputs = model(mfccs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total != 0:
           result = correct / total
           return result * 100
        raise ValueError("数据集为空集")

    # 接下来可以写训练逻辑
    def train(self, validate_data: DataLoader, validate_step: int, model_path: str, latest_steps: int):
        try:
            self.log.info("开始训练")
            self.model.to(self.device)
            self.model.train()

            self.progress.start()
            for epoch in range(self.num_epochs - latest_steps):
                running_loss = 0.0
                for inputs, labels in self.data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 梯度清零
                    self.optimizer.zero_grad()

                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # 添加L2正则化损失
                    l2_reg_loss = self.model.l2_regularization_loss()
                    loss += l2_reg_loss

                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                # validate_step代表多少步验证一次
                if (epoch + 1) % validate_step == 0:
                    self.log.info(f'验证中...')
                    accuracy = self.Accuracy(self.model, data_loader=validate_data, device=self.device)
                    self.log.info(f'准确率: {accuracy}%')
                self.progress.update(self.task, completed=latest_steps + epoch + 1, total=self.num_epochs,
                                    description=f"[cyan]训练中,训练了:{latest_steps + epoch + 1}/{self.num_epochs}")
                self.log.info(f'步数 {latest_steps + epoch + 1}, 损失率: {running_loss / len(self.data_loader.dataset)}')
                # 保存
                if (epoch + 1) % Config.log_interval == 0:
                    trainer.save_model(model_path, latest_steps + epoch + 1)

            self.log.info('训练完成')
        except Exception as e:
            self.log.error("训练的时候发生错误")
            raise RuntimeError("训练时发生错误") from e
        except KeyboardInterrupt:
            self.log.info("训练被中断")
            exit()

        
    # 模型肯定要保存，不然白训练了
    @retry(max_attempts=5,delay=1,backoff=2)
    def save_model(self,model_path:str,steps:int):
        try:
            self.log.info("保存模型")
            save_path = f"./model/{name}/CharacterClassify_{steps}.pth"
            torch.save(self.model.state_dict(), save_path)
            js_path = f"./model/{name}/info.json"
            js_info = {"model":{"model_steps": steps, "model_device": Config.device},}
            with open(js_path, "w", encoding="utf-8") as f:
                json.dump(js_info, f, ensure_ascii=False, indent=2)
            self.log.info("模型保存成功")
        except Exception as e:
            self.log.error("保存模型的时候发生错误")
            raise

    # 从一个模型继续训练
    def train_continue_with_model(self, model_path: str):
        try:
            self.log.info(f"加载已有模型并继续训练，上次模型步数{latest_steps}")
            self.model.load_state_dict(torch.load(model_path))
            self.train(validate_data=validate_data,validate_step=Config.validate_step,model_path=model_path,latest_steps=latest_steps)
        except Exception as e:
            self.log.error("加载模型并继续训练时发生错误")
            raise
        except KeyboardInterrupt:
            self.log.info("用户手动停止训练的继续")

if __name__ == '__main__':
    name = Config.project_name
    
    #上次最后模型路径
    if Config.train_countinue:
       with open(f"./model/{name}/info.json", "r") as rf:
           js_data = json.load(rf)
       latest_steps = js_data.get("model").get("model_steps")
    else:
       latest_steps = 0
    model_path = f"./model/{name}/CharacterClassify_{latest_steps}.pth"
    

    getpath = GetDataPath()
    wav_list ,label_list = getpath.GetPath("train",name)
    val_wav , val_label = getpath.GetPath("validate",name)
    conf = Config("train")
    transform = AudioAugmentation(max_shift=conf.max_shift,noise_factor=conf.noise_factor)
    data_loader = DataLoader(AudioDataset(wav_list,label_list,conf.sr,3,transform),batch_size=conf.batch_size,shuffle=True)
    model = CNNclassifyModel(num_classes=conf.num_classes)
    trainer = Trainer(model=model,data_loader=data_loader,batch_size=conf.batch_size,learning_rate=conf.learning_rate,num_epochs=conf.num_epochs,device=conf.device)
    validate_data = DataLoader(AudioDataset(val_wav,val_label,conf.sr,3,transform=transform),batch_size=conf.batch_size,shuffle=False)
    
    
    if conf.train_countinue:
        trainer.train_continue_with_model(model_path)
    else:
        trainer.train(validate_data=validate_data,validate_step=conf.validate_step,model_path=model_path,latest_steps=0)

