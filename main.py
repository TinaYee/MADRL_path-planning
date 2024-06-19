import argparse
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()

total_r = []

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))
        obs = env.reset()  # 每一回合开始重置状态
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor

        # print(config.env_id)
        # print(env.agent_types)
        # print(obs)
        # print(obs.shape)

        maddpg.prep_rollouts(device='cpu')

        # 根据探索百分比来动态调整噪声的大小
        # 通过线性插值，随着探索的进行，噪声的大小逐渐从初始值过渡到最终值
        # explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        explr_pct_remaining = max(0, config.n_episodes/2 - ep_i) / config.n_episodes/2
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale)
                           * explr_pct_remaining)
        maddpg.reset_noise()

        step_r = []
        for et_i in range(config.episode_length):
            r_all = 0
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # torch_agent_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            # print("rewards: ", rewards)
            # for i in range(maddpg.nagents):
            #     r_all += rewards[0][i]

            step_r.append(rewards[0][0])
            # 每一step的reward为所有agents的reward之和（每一个agent的reward都是所有agent的reward之和）
            # 每一episode的reward为所有steps的reward组成的列表

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

            # 本次回合结束 or 满足终止条件 [:]表示选取整个数组
            if et_i == config.episode_length - 1 or dones.all():
                print(et_i, dones)  # 打印回合结束时的步数 以及 终止状态
                if len(total_r) == 0:
                    total_r.append(np.average(step_r))  # 存储该回合的reward平均值
                else:
                    total_r.append(0.9 * total_r[-1] + 0.1 * np.average(step_r))

                print('Episode:', ep_i, '\t|\t', 'Reward:', total_r[-1],
                      '\t|\tMin:', np.min(total_r), '\t|\tMax:', np.max(total_r))
                break

        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)

        # print("ep_rews: ", ep_rews)

        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


def save_plot():  # 保存数据&画图
    # 将每一回合的reward存为csv文件
    file_name1 = 'E:/lu/orientation/code/maddpg-pytorch-master/results/reward_0.csv'
    save_data(file_name1, total_r)
    # file_name2 = 'E:/lu/orientation/code/maddpg-pytorch-master/results/reward_e.csv'
    # save_data(file_name2, self.evaluate_rewards)

    # 绘制reward曲线
    plt.figure('MADDPG Reward')
    plt.plot(total_r, color='g', linewidth=3)
    plt.xlabel('Episode')
    plt.ylabel('Sum Reward per Episode')
    plt.grid(True)
    plt.show()


def save_data(file_name, datas):
    test = pd.DataFrame(data=datas)
    test.to_csv(file_name, encoding='gbk')
    print("保存文件成功，处理结束")


# 设置训练模型存储路径、算法超参数等
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_id", default="simple_spread", help="Name of environment")
    parser.add_argument("--env_id", default="UAV_path", help="Name of environment")
    parser.add_argument("--model_name",
                        default="E:\lu\orientation\code\maddpg-pytorch-master\model\path",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=7, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=5000, type=int)  # 25000
    parser.add_argument("--episode_length", default=2000, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.35, type=float)  # 0.3
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    # parser.add_argument("--lr_a", default=0.001, type=float)
    # parser.add_argument("--lr_c", default=0.001, type=float)

    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
    save_plot()

