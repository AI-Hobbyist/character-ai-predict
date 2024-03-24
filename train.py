from Log4p.core import *
from config import Config
from Getdata import AudioDataset,AudioAugmentation
from GetPath import GetDataPath
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import global_exc_handler
from rich.progress import track
import torch,os
from Decorators import retry

# 定义CNN游戏角色语音分类模型
class CNNclassifyModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNclassifyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.adapt_pool = nn.AdaptiveAvgPool2d((5, 5))  # 使用自适应池化层替代固定大小的全连接层
        
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 输入大小根据自适应池化层输出的大小确定
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adapt_pool(x)  # 使用自适应池化层
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x

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
           return result
        return 0

    # 接下来可以写训练逻辑
    def train(self,validate_data:DataLoader,validate_step:int,save_epoth_with_model:int):
        try:
            self.log.info("开始训练")
            self.model.to(self.device)
            self.model.train()

            for epoch in track(range(self.num_epochs),description="训练中..."):
                running_loss = 0.0
                for inputs, labels in self.data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # 梯度清零
                    self.optimizer.zero_grad()

                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                # validate_step代表多少步验证一次
                if (epoch+1) % validate_step == 0:
                    self.log.info(f'验证中...')
                    accuracy = self.Accuracy(self.model,data_loader=validate_data,device=self.device)
                    self.log.info(f'准确率: {accuracy}')
                # 设置多少步后保存一次模型:
                if (epoch+1) % save_epoth_with_model == 0:
                    self.log.info(f'保存模型...')
                    self.save_model(f"{Config.save_model_path}/{Config.model_name}_{epoch+1}.pth")
                    self.log.info(f'模型已保存,轮数{epoch+1}')
                self.log.info(f'轮数 {epoch+1}, 损失率: {running_loss/len(self.data_loader.dataset)}')

            self.log.info('训练完成')
        except Exception as e:
            self.log.error("训练的时候发生错误")
            raise RuntimeError("训练时发生错误") from e
        
    # 模型肯定要保存，不然白训练了
    @retry(max_attempts=5,delay=1,backoff=2)
    def save_model(self,model_path:str):
        try:
            self.log.info("保存模型")
            torch.save(self.model.state_dict(), model_path)
            self.log.info("模型保存成功")
        except Exception as e:
            self.log.error("保存模型的时候发生错误")
            raise

    # 从一个模型继续训练
    def train_continue_with_model(self,validate_data:DataLoader,validate_step:int,save_epoth_with_model:int, model_path: str):
        try:
            self.log.info("加载已有模型并继续训练")
            self.model.load_state_dict(torch.load(model_path))
            self.train(validate_data=validate_data,validate_step=validate_step,save_epoth_with_model=save_epoth_with_model)
        except Exception as e:
            self.log.error("加载模型并继续训练时发生错误")
            raise

if __name__ == '__main__':
    getpath = GetDataPath()
    wav_list ,label_list = getpath.GetPath("train")
    val_wav , val_label = getpath.GetPath("validate")
    transform = AudioAugmentation(max_shift=Config.max_shift,noise_factor=Config.noise_factor)
    data_loader = DataLoader(AudioDataset(wav_list,label_list,Config.sr,3,transform),batch_size=Config.batch_size,shuffle=True)
    model = CNNclassifyModel(num_classes=Config.num_classes)
    trainer = Trainer(model=model,data_loader=data_loader,batch_size=Config.batch_size,learning_rate=Config.learning_rate,num_epochs=Config.num_epochs,device=Config.device)
    validate_data = DataLoader(AudioDataset(val_wav,val_label,Config.sr,3,transform=transform),batch_size=Config.batch_size,shuffle=False)
    if Config.train_continue:
        trainer.train_continue_with_model(validate_data=validate_data,validate_step=Config.validate_step,save_epoth_with_model=Config.save_epoth_with_model,model_path=Config.use_model_path)
    else:
        trainer.train(validate_data=validate_data,validate_step=Config.validate_step,save_epoth_with_model=Config.save_epoth_with_model)
    trainer.save_model(f"{Config.save_model_path}/{Config.model_name}_main.pth")
    trainer.log.warning("UserWarning:注意,重新加载模型开始时,不会从上次开始的轮数积累,也就是说模型名称可能会相同,请提前备份好模型,防止被覆盖")