from gym.envs.registration import register
from copy import deepcopy

from . import datasets

register(
    id="rltsp-v26",
    entry_point="gym_rltsp.envs:RltspEnv",
    kwargs={
        "df": deepcopy(datasets.TEST_DATASET),
        #else params
    }
)