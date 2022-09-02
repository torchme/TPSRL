import pandas as pd
import gym
import numpy as np
import argparse
import ray
import ray.rllib as rllib
from ray import tune
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()
parser.add_argument("--train-iterations", type=int, default=10)

# Configs
CONFIG = {
 'lambda_reward': 1.35 #Лямбда число, для увелечения штрафования за попадания на главную диагональ
}


def env_creator(env_config):
    return TpsEnv(env_config)  # return an env instance

def csv_to_matrix(df):
    df


class TpsEnv(gym.Env):
    """


    """

    def __init__(self, env_config, df, first_step_index=0):
        #super(TpsEnv, self).__init__()
        self.df = df.values
        self.min_steps = self.df.shape[0]
        self.df_reward = self._dataframe_to_reward_matrix(self.df)
        self.first_step_index = first_step_index

        self.action_space = gym.spaces.Discrete(self.min_steps-1)
        self.observation_space = gym.spaces.Dict({
            'visited': gym.spaces.MultiBinary(self.min_steps-1),
            'last': gym.spaces.Discrete(self.min_steps)})

        #self.action_space = <gym.Space>
        #self.observation_space = <gym.Space>

    def step(self, action):
        if self.state['visited'][action] == 1:
            self.reward = -10
        else:
            self.state['visited'][action] = 1
            self.reward = -1 * self.df_reward[self.state['last'], action + 1]
            self.state['last'] = action + 1
        if np.all(self.state['visited'] == 1):
            self.reward += -1 * self.df_reward[action + 1, 0]
            self.done = True
        else:
            self.done = False

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.state = {'visited': np.zeros(self.self.min_steps), 'last': 0}
        visited = np.zeros(self.self.min_steps)
        return self.state


    def _dataframe_to_reward_matrix(self, df):
        return np.eye(self.min_steps) * CONFIG['lambda_reward'] * np.max(df) + df



if __name__ == '__main__':


    args = parser.parse_args()
    #ray.init()

    config = rllib.agents.ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "torch"
    config["env_config"] = {}


    ddf = pd.read_csv('../data/raw/dist_vologda_matrix.csv', index_col='Unnamed: 0')
    env = TpsEnv(env_config=config, df=ddf)

    agent = rllib.agents.ppo.PPOTrainer(config=config, env=TpsEnv)
    obs = env.reset()
    #register_env("tps_env", env_creator)

    #tune.run(
    #    config=config,
    #    resources_per_trial= rllib.agents.ppo.PPOTrainer.default_resource_request(config))

    #ray.shutdown()