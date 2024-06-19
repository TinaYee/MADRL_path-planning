import math
import numpy as np
from multiagent.core_aoi_e import World, Agent, Landmark, Satellite
from multiagent.scenario import BaseScenario


# 加入卫星 data rate的计算考虑IoT-无人机-卫星的链路
# 地面设备的位置坐标在每一回合都一样 完全均匀分布（数量需要是平方数）
# 无人机的起点固定在(0, width/2) 终点固定在(length, width/2)
# 优化目标：AoI + energy


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

        # 均匀分布
        x = []
        y = []
        n = int(np.sqrt(len(world.landmarks)))
        for i in range(n):
            for j in range(n):
                x.append((world.width / n * i + world.width / n * (i + 1)) / 2)
                y.append((world.length / n * j + world.length / n * (j + 1)) / 2)

        # 随机分布 5 设备
        # x = [117, 48, 171, 76, 20]
        # y = [12, 87, 90, 149, 172]

        # 10 设备  300m
        # x = [148, 15, 146, 242, 256, 205, 131, 65, 258, 84]
        # y = [83, 120, 270, 215, 122, 68, 175, 189, 39, 101]

        # 10 设备 500m
        x = [365, 165, 159, 81, 432, 372, 487, 261, 57, 323]
        y = [427, 135, 370, 485, 92, 217, 425, 277, 301, 75]

        # data_type = [0, 1, 1, 0, 0]

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

            landmark.state.p_pos = (x[i], y[i], 0)
            # 随机生成设备产生的数据类型 0——时延敏感 1——时延容忍
            # landmark.state.type = np.random.randint(0, 2)
            # landmark.state.type = data_type[i]  # 固定设备数据类型

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.AoI = 0  # 每一回合开始都将总的数据速率初始化为0
        world.energy = 0  # 每一回合开始都将总的能耗初始化为0
        world.all_conn = False

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  # 位置 -1-1随机
            # agent.state.p_vel = np.zeros(world.dim_p)  # 速度 坐标维度为3 初始化为包含x,y,z坐标速度的全0数组
            agent.state.p_vel = 0
            agent.state.c = np.zeros(world.dim_c)  # 通信信息初始化为0 包括无人机选择调度的设备以及是否成功调度
            agent.state.carry = np.zeros(len(world.landmarks))  # 无人机存储携带的数据 初始化为0 携带谁的数据 谁就置为1

            # 随机生成 x 和 y 坐标  [:2]表示取前两个元素
            # agent.state.p_pos[:2] = np.random.uniform(0, world.length / 10, 2)
            # agent.state.p_pos[-1] = np.random.uniform(agent.H_min, agent.H_max)  # 随机生成 z 坐标
            agent.state.p_pos[-1] = (agent.H_min + agent.H_max) / 2  # 固定飞行高度
            agent.state.p_pos[:2] = (0, world.width/2)  # 无人机飞行起点
            agent.state.is_des = False

        data_type = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        for i, landmark in enumerate(world.landmarks):  # 内置枚举函数 i为下标 landmark为列表元素
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  # 地标位置也-1-1之间随机
            landmark.state.p_vel = np.zeros(world.dim_p)  # 速度为0
            # landmark.state.p_pos = np.random.randint(0, world.length, world.dim_p)  # x,y坐标在考虑区域内随机生成
            # landmark.state.p_pos[-1] = 0  # 设置 z 坐标为 0
            landmark.state.is_connected = 0  # 初始状态 均未被调度过
            landmark.state.aoi = 1  # 所有设备的aoi在回合开始时 均初始化为1
            landmark.state.type = data_type[i]  # 固定设备数据类型

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

    def compute_e_fly(self, agent, v_UAV):  # 无人机的飞行能耗
        # e_fly = agent.k1 * pow(v_UAV, 3) + agent.k2 / v_UAV  # first  // 固定翼

        e_fly = agent.P0 * (1 + 3 * pow(v_UAV, 2) / pow(agent.u_tip, 2))\
                + agent.P1 * pow(np.sqrt(1 + pow(v_UAV, 4) / (4 * pow(agent.v0, 4)))
                                 - pow(v_UAV, 2) / (2 * pow(agent.v0, 2)), 1/2)\
                + 1 / 2 * (agent.d0 * agent.rou * agent.s0 * agent.Ar * pow(v_UAV, 3))  # 螺旋翼
        return e_fly


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        # 无人机与终点站之间的距离作为惩罚项
        dis_max = np.sqrt(world.length ** 2 + world.width ** 2)
        delta_pos = agent.state.p_pos[: 2] - agent.des
        dis = np.sqrt(np.sum(np.square(delta_pos)))
        # if not agent.state.is_des:
        #     rew -= dis / dis_max   # 0-1之间

        # 无人机抵达终点后 不再受到奖励 除了所有无人机抵达终点时 all_des 的奖励
        if not agent.state.is_des:
            r_aoi = 0
            # agent 选择调度设备的当前AoI
            for i, l in enumerate(world.landmarks):
                if i == agent.state.c[0]:
                    if agent.state.c[1] == 1:  # 调度成功
                        n_conn = 0
                        for l in world.landmarks:
                            if l.state.is_connected == 1:
                                n_conn += 1
                        # rew += n_conn / len(world.landmarks)
                        rew += n_conn

                        # 如果是中继转发的传输模式 直接获得其AoI
                        if l.state.type == 0:
                            r_aoi += l.state.aoi

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

            # rew -= agent.r_time  # 时间惩罚 鼓励agent尽快完成数据收集任务

            # 根据完成调度设备的数量给出相应奖励
            # n_conn = 0
            # for l in world.landmarks:
            #     if l.state.is_connected == 1:
            #         n_conn += 1
            # rew += n_conn / len(world.landmarks)

            # for a in world.agents:  # 有新的设备被调度 才获得奖励
            #     if a.state.c[1] == 1:
            #         n_conn = 0
            #         for l in world.landmarks:
            #             if l.state.is_connected == 1:
            #                 n_conn += 1
            #         # rew += n_conn / len(world.landmarks)
            #         rew += n_conn
            #         break

            # 无人机抵达目的站 获得 个体奖励
            delta_pos = agent.state.p_pos[: 2] - agent.des
            dis = np.sqrt(np.sum(np.square(delta_pos)))
            if dis <= world.length / 20:
                rew += agent.r_des / len(world.agents)
                agent.state.is_des = True

                # 存储携带的数据的AoI 在无人机抵达终点时再计算其AoI
                for i, l in enumerate(world.landmarks):
                    if agent.state.carry[i] == 1:  # 查询无人机当前存储携带的数据 获取其AoI
                        r_aoi += l.state.aoi

            world.AoI += r_aoi  # 所有数据的AoI之和

            # if r_aoi != 0:
            #     rew += 2 / r_aoi

            max_r_e = self.compute_e_fly(agent, agent.v_max)
            r_e = self.compute_e_fly(agent, agent.state.p_vel)  # 计算无人机在该时隙的飞行能耗
            # rew -= r_e / max_r_e  # 作为惩罚项
            world.energy += r_e

            rew -= (r_aoi / world.landmarks[0].aoi_max + 1e-3 * r_e)
            # rew -= 1e-2 * 0.3 * r_e

            # 数据采集任务完成之后 只获得一次奖励 在无人机抵达终点之前 不反复获得该奖励
            # if not world.all_conn:
            #     # 完成所有设备的数据收集 获得奖励
            #     all_connected = True
            #     for l in world.landmarks:
            #         if l.state.is_connected != 1:
            #             all_connected = False
            #             break
            #     if all_connected:
            #         rew += agent.r_over
            #         world.all_conn = True

        all_des = True
        for a in world.agents:
            if a.state.is_des is False:
                all_des = False
                break
        # if all_des and all_connected:
        if all_des:
            rew += agent.r_des

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        entity_pos = []
        aoi = []  # 当前时刻所有设备的aoi
        for l in world.landmarks:  # world.entities:
            # delta_pos = agent.state.p_pos - l.state.p_pos
            delta_pos = agent.state.p_pos[: 2] - l.state.p_pos[: 2]
            dis = np.sqrt(np.sum(np.square(delta_pos)))
            entity_pos.append(dis)  # 与所有UEs的距离信息
            # entity_pos.append(l.state.p_pos[:2])  # 所有UE的位置信息
            aoi.append(l.state.aoi)

        delta_pos = agent.state.p_pos[: 2] - agent.des
        dis_des = np.sqrt(np.sum(np.square(delta_pos)))

        # communication of all other agents
        comm = []  # 存储所有UE的状态（是否已被调度过）
        for l in world.landmarks:
            comm.append(l.state.is_connected)

        other_pos = []  # 当前智能体与其他智能体之间的距离
        for other in world.agents:
            if other is agent: continue
            delta_pos = agent.state.p_pos - other.state.p_pos
            dis = np.sqrt(np.sum(np.square(delta_pos)))
            other_pos.append(dis)

        obs = np.hstack((np.ravel(agent.state.p_pos), np.ravel(entity_pos), np.ravel(dis_des),
                         np.ravel(other_pos), np.ravel(comm), np.ravel(agent.state.carry),
                         np.ravel(aoi), np.ravel(agent.state.p_vel)))
        return self.obs_normal(agent, world, obs)
        # 观测空间：当前agent的位置、与所有UE的距离、与其他无人机的距离、与终点的距离、
        # 无人机当前携带的数据列表、UE的调度状态、UE的aoi、无人机的速度
        # 观测空间维度为 3 + 3 * n_UEs + n_UAVs - 1 = n_UAVs + 3 * n_UEs + 2

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
            elif 3 + len(world.landmarks) + len(world.agents) - 1 >= i >= 3:  # 与所有UE的距离、其他无人机的距离、与终点的距离
                # max_dis = np.sqrt(world.length ** 2 + world.width ** 2 + agent.H_max ** 2)
                max_dis = np.sqrt(world.length ** 2 + world.width ** 2)
                obs[i] /= max_dis
            elif len(obs) - 2 >= i >= len(obs) - len(world.landmarks) - 1:  # UE的aoi
                obs[i] /= world.landmarks[0].aoi_max
            elif i == len(obs) - 1:  # 无人机的速度
                obs[i] /= agent.v_max
            else:  # 无人机当前携带的数据列表、UE的调度状态
                obs[i] = obs[i]
        return obs

    # 设置回合结束的条件——无人机到达目的站
    def done(self, agent, world):
        all_connected = True
        for l in world.landmarks:
            if l.state.is_connected != 1:
                all_connected = False
                break
        # return agent.state.is_des and all_connected
        return agent.state.is_des
