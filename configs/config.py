import yaml
from yaml.loader import SafeLoader

def make_config():
    yml = """
---
  INPUT_DATA_DIR: data/raw/dist_vologda_matrix.csv
  OUTPUT_DATA_DIR: data/processed/processed.csv
  num_iter: 31
  num_sample_df: 222
... 
"""
    data = yaml.load(yml, Loader=SafeLoader)
    with open('test.yaml', 'w') as fw:
        # сериализуем словарь `user` в формат YAML
        # и записываем все в файл `test.yaml`
        yaml.dump(data, fw, sort_keys=False,
                  default_flow_style=False)

if __name__ == "__main__":
    make_config()
