import numpy as np
import gym
from ray.rllib.env.env_context import EnvContext

class RltspEnv(gym.Env):
    """
Кастомное окружение библиотеки gym.Env

functions
--------
__init__ : Инициализация параметров класса

reset : Функция перезагрузки параметров

step : Функция одного шага

    """
    import numpy as np

    def __init__(self, config: EnvContext, df, matrix_shape=15):

        self.value_matrix = df.values
        self.matrix_shape = matrix_shape
        self.count_steps = self.matrix_shape
        self.n = self.matrix_shape-1

        self.action_space = gym.spaces.Discrete(self.n)
        self.observation_space = gym.spaces.Dict({
            'visited': gym.spaces.MultiBinary(self.n),
            'last': gym.spaces.Discrete(self.count_steps),
            'matrix': gym.spaces.MultiDiscrete(self.value_matrix)
        })


    def reset(self):
        self.state = {'visited': np.zeros(self.n), 'last': 0, 'matrix': self.value_matrix}
        visited = np.zeros(self.n)
        return self.state

    def step(self, action):
        if self.state['visited'][action] == 1:
            self.reward = -1* np.max(self.value_matrix)
            print(f' reward1: {self.reward}')
        else:
            self.state['visited'][action] = 1
            self.reward = -1* self.value_matrix[self.state['last'], action + 1]
            self.state['last'] = action + 1
            print(f' reward2: {self.reward}')
        if np.all(self.state['visited'] == 1):
            self.reward += -1* self.value_matrix[action + 1, 0]
            self.done = True
        else:
            self.done = False

        return self.state, self.reward, self.done, {}
