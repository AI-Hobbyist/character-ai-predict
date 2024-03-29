import os
from Log4p.core import *



class GetDataPath:
    def __init__(self):
        self.log = LogManager().GetLogger("GetDataPath")
        pass

    def GetPath(self, base_name: str, name: str) -> tuple[list, list]:
        dataset_path = f"datasets/{name}/{base_name}"
        #检测文件夹datasets/name存不存在,如果不存在,则创建一个
        if not os.path.exists(f"datasets/{name}"):
            self.log.warning(f"文件夹 datasets/{name} 不存在,自动创建")
            os.makedirs(f"datasets/{name}",exist_ok=True)
        
        class_data = {}  # 使用字典来存储每个类别的数据
        min_data_count = float('inf')  # 记录最小数据集的样本数量
        
        for root, dirs, files in os.walk(dataset_path):
            for label in dirs:
                label_path = os.path.join(root, label)
                class_data[int(label)] = []
                num_samples = 0
                for wav_file in os.listdir(label_path):
                    wav_path = os.path.join(label_path, wav_file)
                    class_data[int(label)].append(wav_path)
                    num_samples += 1
                min_data_count = min(min_data_count, num_samples)
        
        if min_data_count == 0:
            return [], []
        
        final_wav_list = []
        final_label_list = []
        
        while True:
            all_empty = True
            for label, data_list in class_data.items():
                if len(data_list) > 0:
                    all_empty = False
                    final_wav_list.append(data_list.pop(0))
                    final_label_list.append(label)
                    if len(final_wav_list) >= min_data_count * len(class_data):
                        break
            if all_empty or len(final_wav_list) >= min_data_count * len(class_data):
                break
        
        return final_wav_list, final_label_list

if __name__ == "__main__":
    d = GetDataPath()

    wav_list , label_list = d.GetPath("train","芙宁娜")
    print(wav_list)
    print(label_list)