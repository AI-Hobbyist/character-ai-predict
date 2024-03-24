import os



class GetDataPath:
    def __init__(self):
        pass

    def GetPath(self,base_name:str) -> tuple[list,list]:
        wav_list = []
        label_list = []
        dataset_path = f"datasets/{base_name}"
        for root, dirs, files in os.walk(dataset_path):
            for label in dirs:
                label_path = os.path.join(root, label)
                for wav_file in os.listdir(label_path):
                    wav_path = os.path.join(label_path, wav_file)
                    wav_list.append(wav_path)
                    label_list.append(int(label))
        return wav_list,label_list