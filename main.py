import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from environment.env import Environment

if __name__ == '__main__':
    mp.set_start_method("spawn")

    env = Environment()
    print("Running environment...")
    env.run()
