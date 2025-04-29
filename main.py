import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from environment.env import Environment

if __name__ == '__main__':
    mp.set_start_method("spawn")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:,.5f}'.format
    
    
    env = Environment()
    print("Running environment...")
    env.run() 