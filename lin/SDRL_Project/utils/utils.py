# a set of helper functions
# -----------------------------
import torch
import numpy as np
import random


# noinspection PyBroadException
def set_seed_everywhere(seed, env):
    print('[INFO] Setting random seed to ', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'seed' in dir(env):
        try:
            env.seed(seed)
        except Exception:
            pass
