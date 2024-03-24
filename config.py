

class Config:
    model_name = "CharacterClassify" #模型名字
    validate_step = 5 #每多少步验证一次
    num_epochs = 10 # 训练的总轮数
    batch_size = 16 # 训练抽取样本多少条
    save_epoth_with_model = 5 # 每多少步保存一次模型
    learning_rate = 1e-6 # 学习率
    device = "cuda" # 使用什么设备训练
    num_classes = 60 # 需要分类多少个角色
    save_model_path = "./model" # 保存模型的路径
    sr = 44100 # 采样率
    max_shift = 1000 # 不知道
    noise_factor = 0.05 # 不知道
    train_continue = False # 是否是继续训练的模式
    use_model_path = "./model/Character.pth" # 如果是继续训练的模式,则填入这个使用的模型文件路径
    