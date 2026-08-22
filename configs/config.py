# 标准库
import os.path

import yaml

# 内部工具
from utils.path import PROJECT_PATH


# 读入yaml文件
config_path = os.path.join(PROJECT_PATH, "configs/config.yaml")
with open(config_path, "r") as file:
    configs = yaml.safe_load(file)
