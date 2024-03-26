import yaml

class Config:
    def __init__(self,base_name:str) -> None:
        self.yaml_conf = self.load_yaml(base_name=base_name)

    def load_yaml(self,base_name:str):
        with open("config.yaml", "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        return conf[base_name]
    
    @property
    def validate_step(self) -> int:
        return self.yaml_conf['validate_step']
    
    @property
    def num_epochs(self) -> int:
        return self.yaml_conf['num_epochs']
    
    @property
    def batch_size(self) -> int:
        return self.yaml_conf['batch_size']
    
    @property
    def learning_rate(self) -> float:
        return self.yaml_conf['learning_rate']
    
    @property
    def device(self) -> str:
        return self.yaml_conf['device']
    
    @property
    def num_classes(self) -> int:
        return self.yaml_conf['num_classes']
    
    @property
    def project_name(self) -> str:
        return self.yaml_conf['project_name']
    
    @property
    def sr(self) -> int:
        return self.yaml_conf['sr']
    
    @property
    def max_shift(self) -> int:
        return self.yaml_conf['max_shift']
    
    @property
    def noise_factor(self) -> float:
        return self.yaml_conf['noise_factor']
    
    @property
    def train_countinue(self) -> bool:
        return self.yaml_conf['train_countinue']
    
    @property
    def log_interval(self) -> int:
        return self.yaml_conf['log_interval']