#from configs.config import Config
from src.data.make_dataset import read_csv
from configs.config import make_config
from src.models.rl_tps import main

import yaml
from yaml.loader import SafeLoader

if __name__ == "__main__":
    print('Making DataSet & config')
    make_config()
    with open('configs/test.yaml', 'r') as fr:
        config = yaml.load(fr.read(), Loader=SafeLoader)
    matrix_shape, value_matrix, df = read_csv(config['INPUT_DATA_DIR'], config['OUTPUT_DATA_DIR'], 'Unnamed: 0')

    print(config)
    print('Finish')


    main(matrix_shape, value_matrix, df)