import math
import numpy as np
from multiagent.core_ee import World, Agent, Landmark, Satellite
from multiagent.scenario import BaseScenario


# 加入卫星 data rate的计算考虑IoT-无人机-卫星的链路
# 无人机的起点固定在原点处
# add 未正确调度的惩罚

# 加入无人机能耗


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        num_agents = 2  # 无人机数量
        num_landmarks = 10  # 地面UE数量
        num_satellites = 1  # 卫星数量
        world.dim_c = 2  # 通信动作维度——当前选择调度的设备编号、是否为首次调度

        world.satellites = [Satellite() for i in range(num_satellites)]  # 初始化卫星对象

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

            # 设置动作噪声大小
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.data_rate = 0  # 每一回合开始都将总的数据速率初始化为0
        world.energy = 0  # 每一回合开始都将总的能耗初始化为0

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  # 位置 -1-1随机
            agent.state.p_vel = 0  # 速度 坐标维度为3 初始化为包含x,y,z坐标速度的全0数组
            agent.state.c = np.zeros(world.dim_c)  # 通信信息初始化为0

            # 随机生成 x 和 y 坐标  [:2]表示取前两个元素
            # agent.state.p_pos[:2] = np.random.uniform(0, world.length / 10, 2)
            # agent.state.p_pos[-1] = np.random.uniform(agent.H_min, agent.H_max)  # 随机生成 z 坐标
            agent.state.p_pos[-1] = (agent.H_min + agent.H_max) / 2  # 固定飞行高度

            agent.state.p_pos[:2] = (0, 0)  # 圆点为无人机飞行起点

        # x = [117, 48, 171, 76, 10]
        # y = [12, 87, 90, 149, 192]

        # 10 设备  300m
        x = [46, 242, 284, 109, 142, 280, 132, 209, 58, 219]
        y = [231, 205, 54, 168, 265, 297, 84, 102, 45, 270]

        # 10 设备 500m
        # x = [365, 165, 159, 81, 432, 372, 487, 261, 57, 323]
        # y = [427, 135, 370, 485, 92, 217, 425, 277, 301, 75]

        for i, landmark in enumerate(world.landmarks):  # 内置枚举函数 i为下标 landmark为列表元素
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  # 地标位置也-1-1之间随机
            landmark.state.p_vel = np.zeros(world.dim_p)  # 速度为0
            # landmark.state.p_pos = np.random.randint(0, world.length, world.dim_p)  # x,y坐标在考虑区域内随机生成
            # landmark.state.p_pos[-1] = 0  # 设置 z 坐标为 0
            landmark.state.is_connected = 0  # 初始状态 均未被调度过
            landmark.state.p_pos = (x[i], y[i], 0)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):  # 判断智能体之间是否发生碰撞
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        dist_min = agent1.d_min  # 无人机之间的最小安全距离
        return True if dist < dist_min else False

    def is_out(self, agent, world):
        flag = False
        if agent.state.p_pos[0] < 0 or agent.state.p_pos[0] > world.length \
                or agent.state.p_pos[1] < 0 or agent.state.p_pos[1] > world.width:
            flag = True
        return flag

    def is_repeat_sametime(self, agent, world):
        repeat_schedule = False
        for a in world.agents:  # 同一时隙调度相同设备 且均为首次调度
            if a == agent: continue
            if a.state.c[0] == agent.state.c[0]:
                if a.state.c[1] == agent.state.c[1] == 1:
                    repeat_schedule = True
                    break
        return repeat_schedule

    def is_covered(self, agent, l):
        covered = False
        delta_pos = agent.state.p_pos[:2] - l.state.p_pos[:2]
        dis = np.sqrt(np.sum(np.square(delta_pos)))
        if dis <= agent.state.p_pos[2] * math.tan(math.radians(agent.angle)):
            covered = True
        return covered

    def compute_γ_G2A(self, agent, world, x1, y1, x2, y2, h, P_UE):
        # 计算G2A链路的信噪比
        # UE k和UAV m之间的仰角
        # np.rad2deg 弧度rad转化为角度
        φ = np.rad2deg(np.arctan(h / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)))
        # G2A链路的Los概率
        P_los_G2A = 1 / (1 + world.a * np.exp(-world.b * (φ - world.a)))
        # UE k和UAV m之间的距离
        dis_km = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + h ** 2)
        # G2A链路的路径损耗
        L_G2A = 20 * np.log(4 * math.pi * world.fc * dis_km / world.v) + P_los_G2A * world.η_LOS \
                + (1 - P_los_G2A) * world.η_NLOS
        # G2A链路的信道增益
        g_G2A = 10 ** (-L_G2A / 10)
        # G2A链路的信噪比
        γ_G2A = P_UE * g_G2A / agent.N_UAV
        return γ_G2A

    def compute_γ_A2S(self, agent, satellite, h, P_UAV):
        # 计算A2S链路的信噪比
        # 信道增益  降雨衰减
        g_A2S = agent.G_tx * satellite.G_rx * (satellite.lamda) ** 2 * (10 ** (-satellite.Frain / 10)) \
                / (4 * math.pi * (satellite.Hs - h)) ** 2
        γ_A2S = P_UAV * g_A2S / satellite.N_LEO  # 信噪比
        return γ_A2S

    def compute_r(self, agent, world, satellite, x1, y1, x2, y2, h, P_UE, P_UAV):
        γ_G2A = self.compute_γ_G2A(agent, world, x1, y1, x2, y2, h, P_UE)
        γ_A2S = self.compute_γ_A2S(agent, satellite, h, P_UAV)
        R = world.W * np.log2(1 + (γ_G2A * γ_A2S / (1 + γ_G2A + γ_A2S)))  # 加入UAV-satellite链路
        R *= world.reward_scale
        return R

    def compute_e_fly(self, agent, v_UAV):  # 无人机的飞行能耗
        # e_fly = agent.k1 * pow(v_UAV, 3) + agent.k2 / v_UAV  # first  // 固定翼

        e_fly = agent.P0 * (1 + 3 * pow(v_UAV, 2) / pow(agent.u_tip, 2))\
                + agent.P1 * pow(np.sqrt(1 + pow(v_UAV, 4) / (4 * pow(agent.v0, 4)))
                                 - pow(v_UAV, 2) / (2 * pow(agent.v0, 2)), 1/2)\
                + 1 / 2 * (agent.d0 * agent.rou * agent.s0 * agent.Ar * pow(v_UAV, 3))  # 螺旋翼
        return e_fly

    def compute_e_com(self, world, agent, l, satellite, h, P_UAV):  # 无人机的通信能耗
        γ_A2S = self.compute_γ_A2S(agent, satellite, h, P_UAV)
        c_A2S = world.W * np.log2(1 + γ_A2S)
        e_com = P_UAV * l.data / c_A2S
        return e_com

    # def reward(self, agent, world):
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        r_data = 0

        # agent与选择调度设备之间的数据传输速率
        for i, l in enumerate(world.landmarks):
            if i == agent.state.c[0]:
                if agent.state.c[1] == 1:  # 首次调度 才能获得数据传输速率的奖励
                    r_data = self.compute_r(agent, world, world.satellites[0], l.state.p_pos[0], l.state.p_pos[1],
                                            agent.state.p_pos[0], agent.state.p_pos[1], agent.state.p_pos[2],
                                            l.P_UE, agent.P_UAV_max)
                    # rew += r_data  # 数据传输速率
                    world.data_rate += r_data

                    delta_pos = agent.state.p_pos[:2] - l.state.p_pos[:2]
                    dis = np.sqrt(np.sum(np.square(delta_pos)))
                    dis_max = agent.state.p_pos[2] * math.tan(math.radians(agent.angle))  # 在覆盖范围内的最大距离
                    # rew -= dis / dis_max

        if agent.collide:  # 碰撞惩罚
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(agent, a):
                    rew -= agent.r_collide
                    # print(et_i, "collision!!!")

        if agent.out:  # 飞出边界惩罚
            if self.is_out(agent, world):
                rew -= agent.r_out
                # print(et_i, "out!!!")

        max_r_e = self.compute_e_fly(agent, agent.v_max)
        r_e = self.compute_e_fly(agent, agent.state.p_vel)  # 计算无人机在该时隙的飞行能耗
        # rew -= r_e / max_r_e  # 作为惩罚项
        world.energy += r_e

        rew += (r_data - 1e-2 * 0.1 * r_e)

        # 根据完成调度设备的数量给出相应奖励
        # for a in world.agents:
        #     if a.state.c[1] == 1:
        #         n_conn = 0
        #         for l in world.landmarks:
        #             if l.state.is_connected == 1:
        #                 n_conn += 1
        #         rew += n_conn / len(world.landmarks)
        #         break

        # 完成所有设备的数据收集 获得奖励
        all_connected = True
        for l in world.landmarks:
            if l.state.is_connected != 1:
                all_connected = False
                break
        if all_connected:
            rew += agent.r_over
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for l in world.landmarks:  # world.entities:
            # delta_pos = agent.state.p_pos - l.state.p_pos
            delta_pos = agent.state.p_pos[: 2] - l.state.p_pos[: 2]
            dis = np.sqrt(np.sum(np.square(delta_pos)))
            entity_pos.append(dis)  # 与所有UEs的距离信息
            # entity_pos.append(l.state.p_pos[:2])  # 所有UE的位置信息
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []  # 存储所有UE的状态（是否已被调度过）
        for l in world.landmarks:
            comm.append(l.state.is_connected)

        other_pos = []  # 当前智能体与其他智能体之间的距离
        for other in world.agents:
            # comm.append(other.state.c)  # 改——注释掉
            if other is agent: continue
            delta_pos = agent.state.p_pos - other.state.p_pos
            dis = np.sqrt(np.sum(np.square(delta_pos)))
            other_pos.append(dis)

        obs = np.hstack((np.ravel(agent.state.p_pos), np.ravel(entity_pos),
                        np.ravel(comm), np.ravel(agent.state.p_vel)))
        return self.obs_normal(agent, world, obs)
        # return np.hstack((np.ravel(agent.state.p_pos), np.ravel(entity_pos), np.ravel(other_pos), np.ravel(comm)))
        # 观测空间维度为 3 + 2 * n_UEs + n_UAVs-1 + n_UEs = n_UAVs + 3 * n_UEs + 2

    # 手动对观测空间进行归一化处理
    def obs_normal(self, agent, world, obs):
        for i in range(len(obs)):
            if i == 0:  # agent的x、y坐标
                obs[i] /= world.length
                obs[i + 1] /= world.width
            elif i == 2:  # agent的z坐标
                obs[i] /= agent.H_max
            # elif 3 + len(world.landmarks) * 2 - 1 >= i >= 3:  # 所有UE的位置坐标
            #     obs[i] /= world.length
            elif 3 + len(world.landmarks) - 1 >= i >= 3:  # 与所有UE的距离
                # max_dis = np.sqrt(world.length ** 2 + world.width ** 2 + agent.H_max ** 2)
                max_dis = np.sqrt(world.length ** 2 + world.width ** 2)
                obs[i] /= max_dis  # 与所有UE的距离
            elif i == len(obs) - 1:  # 无人机的当前速度
                obs[i] /= agent.v_max
            else:  # UE的调度状态
                obs[i] = obs[i]
        return obs

    # 设置回合结束的条件——全部飞出边界 or 数据收集任务完成
    def done(self, agent, world):
        doneInfo = []
        if self.is_out(agent, world):
            doneInfo.append(True)
        else:
            doneInfo.append(False)
        all_connected = True
        for i, l in enumerate(world.landmarks):
            if l.state.is_connected != 1:
                all_connected = False
                break
        doneInfo.append(all_connected)
        return all_connected
