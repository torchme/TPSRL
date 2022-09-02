import pandas as pd
import os
name = 'test2'
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, 'data', name+'.csv')
TEST_DATASET = pd.read_csv(path, index_col='Unnamed: 0')