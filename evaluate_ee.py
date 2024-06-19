import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from multiagent.core_ee import World, Agent, Landmark
import pandas as pd

# 初始化 存储UAV和UE位置坐标的数组
UAV_pos0 = []
UAV_pos1 = []
UAV_pos2 = []
UAV_pos3 = []

UAV_pos = []
UE_pos = []

total_r = []

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval


    for ep_i in range(config.n_episodes):
        noise_std = 0

        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()

        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])

        # env.render('human')

        step_r = []
        for t_i in range(config.episode_length):
            r_all = 0
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            # torch_actions = maddpg.step(torch_obs, explore=False)
            torch_actions = maddpg.step(torch_obs, noise_std, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)

            for i in range(maddpg.nagents):
                r_all += rewards[i]
            step_r.append(r_all)  # 每一step的reward 为所有agents的reward之和

            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)

            # env.render('human')
            # env.render('rgb_array')

            # 最后一回合时 保存所有无人机在该回合的飞行轨迹
            if ep_i == config.n_episodes - 1:
                pos = []
                for i, a in enumerate(env.world.agents):
                    pos.extend(np.array(a.state.p_pos))
                    pos.extend(a.state.c)
                UAV_pos.append(pos)

            # 本次回合结束 or 满足终止条件 [:]表示选取整个数组
            if t_i == config.episode_length - 1 or all(dones):
                if ep_i == config.n_episodes -1:  # 最后一回合 存储该回合所有UE的位置坐标
                    for l in env.world.landmarks:
                        UE_pos.append(l.state.p_pos[:2])

                print(t_i, dones)  # 打印回合结束时的步数 以及 终止状态
                if len(total_r) == 0:
                    total_r.append(np.average(step_r))  # 存储该回合的reward值
                    # total_r.append(np.sum(step_r))  # 存储该回合的reward值——为所有steps的reward之和
                else:
                    total_r.append(0.9 * total_r[-1] + 0.1 * np.average(step_r))
                    # total_r.append(0.9 * total_r[-1] + 0.1 * np.sum(step_r))

                print('Episode:', ep_i, '\t|\t', 'Reward:', total_r[-1],
                      'data rate:', env.world.data_rate, 'energy:', env.world.energy,
                      '\t|\tMin:', np.min(total_r), '\t|\tMax:', np.max(total_r))
                break

        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()

def save_plot():  # 保存数据&画图
    # 将每一回合的reward存为csv文件
    file_name1 = 'E:/lu/orientation/code/maddpg-pytorch-master/results/ee/reward_e.csv'
    # save_data(file_name1, total_r)

    # 存储最后一回合的无人机轨迹
    file_name2 = 'E:/lu/orientation/code/maddpg-pytorch-master/results/ee/UAV_path.csv'
    save_data(file_name2, UAV_pos)

    # 存储最后一回合的地面设备位置坐标
    file_name3 = 'E:/lu/orientation/code/maddpg-pytorch-master/results/ee/UE_pos.csv'
    save_data(file_name3, UE_pos)

def save_data(file_name, datas):
    test = pd.DataFrame(data=datas)
    test.to_csv(file_name, encoding='gbk')
    # print("保存文件成功，处理结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="UAV_path_r_ee", help="Name of environment")
    parser.add_argument("--model_name",
                        default="E:\lu\orientation\code\maddpg-pytorch-master\model\path_ee",
                        help="Name of model")
    parser.add_argument("--run_num", default=2, type=int)

    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")

    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=2, type=int)
    parser.add_argument("--episode_length", default=300, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
    save_plot()