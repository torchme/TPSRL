import pandas as pd
import numpy as np
import gym
import ray
import ray.rllib as rllib
from ray import tune
from ray.tune.registry import register_env
import wandb


CONFIG = {
    'pathToDf':'data/',
    'num_iter': 31,
    'num_sample_df': 26
}

def read_csv(df_name='df.csv'):
    """
Функция которая читает датасет и переводит его в numpy матрицу

Parameters
----------
df_name : Название датасета

Returns
-------
matrix_shape : int, Размерность матрицы
value_matrix : numpy list, Матрица значений
df : pandas DataFrame, Считанный датафрейм данных

    """
    df = pd.read_csv(CONFIG['pathToDf'] + df_name, index_col='Unnamed: 0')
    df = df.iloc[:CONFIG['num_sample_df'], :CONFIG['num_sample_df']] # Убрать при деплое, используется для ограничения размерности
    matrix_shape = df.shape[0] # N
    value_matrix = df.values # M

    return matrix_shape, value_matrix, df

def get_inference(agent):
    """
Функция инференса обученного агента

Parameters
----------
agent : rllib.agent, обученный агент для инференса

Returns
-------
None
    """
    total_dist = 0
    actions = [0]
    obs = env.reset()
    for i in range(CONFIG['num_sample_df']):
        action = agent.compute_single_action(obs, explore=False)
        # print(f'action: {action}, action space: {env.action_space}')
        obs, reward, done, info = env.step(action)
        total_dist += value_matrix[actions[-1], action + 1]
        actions.append(action + 1)
        # print(f'{g} was added by {M[actions[-1], action+1]}')
        # print(reward)
        if done:
            obs = env.reset()
            total_dist += value_matrix[actions[-1], 0]
            print(f"Done")
            break
        env.close()

    print(actions)
    print(total_dist)

def train_func(config, checkpoint_dir='agents/ppo_last_checkpoint'):
    """
Функция трейнлупа с интеграцией `wandb.ai` для трекинга результатов и сохранения артефактов.
W&B log получается статистические параметры reward и размер episodes. W&B artefact сохраняет обученного агента.

Parameters
----------
config : class, rllib agent config
checkpoint_dir : str, Директория по которой будет сохраняться агент

Returns
-------
agent : Обученный агент
    """

    run = wandb.init(project=f"TSPRL", entity="torchme")
    wandb.run.name = f"TPSRL-{CONFIG['num_sample_df']}x{CONFIG['num_sample_df']}"

    for _ in range(CONFIG['num_iter']):
        result = agent.train()
        wandb.log({
            'mean episode length:': result['episode_len_mean'],
            'max episode reward:': result['episode_reward_max'],
            'mean episode reward:': result['episode_reward_mean'],
            'min episode reward:': result['episode_reward_min'],
            'total episodes:': result['episodes_total'],
        })
        print('mean episode length:', result['episode_len_mean'])
        print('max episode reward:', result['episode_reward_max'])
        print('mean episode reward:', result['episode_reward_mean'])
        print('min episode reward:', result['episode_reward_min'])
        print('total episodes:', result['episodes_total'])
    agent.save('agents/ppo_last_checkpoint')

    artifact = wandb.Artifact(name='PPOlastCheckPoint', type='artifact_type')
    artifact.add_dir('agents/ppo_last_checkpoint/checkpoint_000031')
    run.log_artifact(artifact)
    run.finish()

    agent.stop()
    return agent

def env_creator(env_config):

    return MyEnv(env_config)  # return an env instance

class MyEnv(gym.Env):
    """
Кастомное окружение библиотеки gym.Env

functions
--------
__init__ : Инициализация параметров класса

reset : Функция перезагрузки параметров

step : Функция одного шага

    """
    def __init__(self, env_config):
        super().__init__()
        self.count_steps = matrix_shape
        self.n = matrix_shape-1
        self.action_space = gym.spaces.Discrete(self.n)
        self.observation_space = gym.spaces.Dict({
            'visited': gym.spaces.MultiBinary(self.n),
            'last': gym.spaces.Discrete(matrix_shape)})

    def reset(self):
        self.state = {'visited': np.zeros(self.n), 'last': 0}
        visited = np.zeros(self.n)
        return self.state

    def step(self, action):
        if self.state['visited'][action] == 1:
            self.reward = -1* np.max(value_matrix)
            print(f' reward1: {self.reward}')
        else:
            self.state['visited'][action] = 1
            self.reward = -1* value_matrix[self.state['last'], action + 1]
            self.state['last'] = action + 1
            print(f' reward2: {self.reward}')
        if np.all(self.state['visited'] == 1):
            self.reward += -1* value_matrix[action + 1, 0]
            self.done = True
        else:
            self.done = False

        return self.state, self.reward, self.done, {}


if __name__ == '__main__':
    ray.init(log_to_driver=False)

    matrix_shape, value_matrix, df = read_csv('dist_vologda_matrix.csv')

    config = rllib.agents.ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["framework"] = "torch"
    config["env_config"] = {}


    env = MyEnv(config)

    agent = rllib.agents.ppo.PPOTrainer(config=config, env=MyEnv)
    obs = env.reset()
    register_env("my_env", env_creator)

    agent = train_func(agent)

    experiment = tune.run(
        rllib.agents.ppo.PPOTrainer,
        config={
            "env": "my_env",
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch"
        },
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 50}
    )

    get_inference(agent)

    ray.shutdown()
