import numpy as np
import math
import seaborn as sns


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        self.is_connected = None  # 调度状态  0——未调度 1——被调度
        self.aoi = None  # AoI
        self.type = None  # 产生的数据类型  0——时延敏感 1——时延容忍


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        self.carry = None
        self.is_des = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True

        self.out = True  # 考虑是否飞出边界
        self.repeat = True  # 考虑是否重复调度

        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.P_UE = 0.8
        self.success_prob = 0.5  # 成功生成新数据包的概率
        self.aoi_max = 20


class Satellite(Entity):
    def __init__(self):
        super(Satellite, self).__init__()
        self.Hs = 5e5  # 500km LEO卫星高度
        self.C_max = 2e8  # 200Mbps 卫星最大容量

        self.G_rx = 1e5  # dB 卫星天线增益
        # 卫星接收器处的噪声功率谱密度 -174 dBm/Hz
        self.N_LEO = 10 ** ((-174 - 30) / 10)
        self.g0 = 1.4e-4  # UAV到LEO卫星参考距离1m处的信道增益
        self.lamda = 8.3e-9  # 载波波长
        self.Frain = 6  # dB weibull模型的降雨衰减


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.H_min = 100
        self.H_max = 150
        self.Δh_max = 5

        self.P_UAV_max = 50  # W UAV的传输功率限制
        self.v_max = 20  # m/s UAV最大飞行速度
        self.d_min = 30  # m UAV之间的安全距离
        self.G_tx = 1e5  # dB UAV天线增益
        # 无人机接收器处的噪声功率谱密度 -169 dBm/Hz
        self.N_UAV = 10 ** ((-169 - 30) / 10)

        self.angle = 15  # 覆盖角

        self.r_collide = 1
        self.r_out = 1
        self.r_repeat = 5
        self.r_first = 0.5
        self.r_schedule = 0.02
        self.r_time = 0.001  # 时间惩罚
        self.r_over = 10
        self.r_des = 10

        self.P0 = 79.86  # 悬停状态下的叶型功率 W
        self.P1 = 88.63  # 悬停状态下的诱导功率
        self.u_tip = 120  # 旋翼桨叶的尖端速度 m/s
        self.v0 = 4.03  # 悬停状态下平均旋翼诱导速度 m/s
        self.d0 = 0.6  # 机身阻力比
        self.rou = 1.225  # 空气密度 kg/m3
        self.s0 = 0.05  # 螺旋翼固体度
        self.Ar = 0.503  # 螺旋翼盘面积 m2

        self.des = np.array([World().length, World().width / 2])  # 无人机终点坐标


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.satellites = []
        self.walls = []
        # communication channel dimensionality 通信信道维度
        self.dim_c = 0
        # self.dim_c = 1
        # position dimensionality  坐标维度  改成三维
        self.dim_p = 3

        self.dim_a = 3  # agent的动作维度（飞行方向、飞行速度、设备调度，暂时不加飞行高度）

        # color dimensionality  颜色维度
        self.dim_color = 3
        # simulation time step   仿真时隙
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        # 所有agents 之间的缓存距离(默认情况下不计算)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

        self.length = 500  # m 所考虑区域的长度
        self.width = 500  # m 宽度 1000m×1000m的方形区域
        self.a = 4.88
        self.b = 0.43  # 环境变量
        self.η_LOS = 0.1  # dB 视距链路的自由空间路径损耗引起的平均附加损耗
        self.η_NLOS = 21  # dB 非视距链路的自由空间路径损耗引起的平均附加损耗

        self.fc = 2e9  # 2GHz 载波频率
        self.v = 3e8  # 3×10^8m/s 光速
        self.T = 20  # s 整个时间过程
        self.N = 20  # t=T/N
        self.t = self.T / self.N  # s 一个时隙
        self.W = 5e6  # 5MHz 系统分配总带宽
        self.reward_scale = 1e-7

        self.AoI = 0  # 初始化该回合的总信息年龄为0
        self.energy = 0  # 初始化该回合的总能耗为0

        self.all_conn = False

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = sns.color_palette("OrRd_d", n_adversaries)
        good_colors = sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # 改——原来更新agent位置的代码注释掉
        # # gather forces applied to entities  初始化长度为环境中所有实体个数的列表
        # p_force = [None] * len(self.entities)
        # # apply agent physical controls  返回向每个智能体施加的物理力（飞行方向、速度...）
        # p_force = self.apply_action_force(p_force)
        # # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        # # integrate physical state  更新agent状态（位置坐标）
        # self.integrate_state(p_force)

        # update agent state  更新agent位置以及通信状态
        for agent in self.agents:
            # all_connected = True
            # for l in self.landmarks:
            #     if l.state.is_connected != 1:
            #         all_connected = False
            #         break
            # if all_connected and agent.state.is_des:  # 完成数据传输任务且到达终点之后就不再更新状态
            if agent.state.is_des:  # 无人机到达终点之后就不再更新状态
                for i, l in enumerate(self.landmarks):
                    if agent.state.carry[i] == 1:
                        l.state.type = 2  # 将无人机携带的数据类型置为2 表示其已被卸载
            else:
                self.update_state(agent)  # 新加

        # 更新所有设备的AoI
        for l in self.landmarks:
            if l.state.is_connected == 1:  # 已被调度 需进一步根据其数据类型来判断其传输模式
                # 若设备的数据类型为0（时延敏感），则说明被中继转发 则AoI不再改变
                # if l.state.type == 0:
                #     continue
                # else:  # 若设备的数据类型为1（时延容忍），则说明被存储携带 则设备的AoI在到达目的站之前持续增加
                #     l.state.aoi = min(l.aoi_max, l.state.aoi + 1)
                if l.state.type == 1:  # 若设备的数据类型为1（时延容忍），说明被存储携带，则设备的AoI在到达目的站之前持续增加
                    l.state.aoi = min(l.aoi_max, l.state.aoi + 1)
                else:  # 数据类型为0或2 AoI都不再改变
                    continue
                # 数据类型为0——（时延敏感），说明被中继转发，AoI不再改变
                # 数据类型为2——说明携带存储的数据抵达目的站，被卸载
            else:
                self.update_aoi(l)

            # self.update_agent_state(agent)  # 改——注释掉 物理和通信动作合并 状态更新也合并
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                # 如果agent.u_noise为真（非零），则生成一个随机噪声向量，否则，将噪声设置为0.0
                # np.random.randn(*agent.action.u.shape): 使用NumPy库中的np.random.randn函数
                # 生成一个与agent.action.u具有相同形状的随机数向量
                # np.random.randn生成的随机数是服从标准正态分布（均值为0，标准差为1）的随机数
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None else agent.mass) \
                             * agent.action.u + noise
        return p_force

    # gather physical forces acting on entities 计算环境中的实体所受的物理力
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a): continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    # integrate physical state 更新环境中物体的物理状态（例如位置和速度）
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            # 根据阻尼系数self.damping减小实体的速度
            # 阻尼模拟了物体在移动过程中可能遇到的摩擦或空气阻力
            # 每个时间步长后，速度会减小一定比例，以模拟物体的逐渐减速
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                # 根据牛顿的第二定律 F = ma，将施加在实体上的力p_force[i]除以实体的质量entity.mass
                # 得到加速度，然后乘以时间步长self.dt来更新速度。这模拟了物体受到力后的加速和速度的变化
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:  # 检查实体是否有最大速度限制
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    # 将速度向量重新缩放，使其大小等于实体的最大速度，以限制速度的上限
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0])
                                                                      + np.square(
                        entity.state.p_vel[1])) * entity.max_speed
            # 速度向量乘以时间步长self.dt来更新实体的位置，这模拟了物体在每个时间步长内的位移
            entity.state.p_pos += entity.state.p_vel * self.dt

    # 主要是为智能体设置通信状态
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            # 生成一个随机噪声向量noise。这个噪声的形状与智能体的动作agent.action.c相同
            # 噪声向量通常用于引入通信中的不确定性或随机性
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
            # 将动作向量agent.action.c与噪声向量相加，以获得最终的通信状态
            # agent.state.c这个操作将动作和噪声结合起来，用于模拟智能体的通信行为，其中通信状态受到动作和噪声的影响

            # 改——根据智能体调度设备的动作 更新UE的状态
            for i, landmark in enumerate(self.landmarks):
                if i == agent.state.c:
                    landmark.state.is_connected = 1

    # 更新所有设备的AoI——如果有新数据包到达，aoi置为1，否则+1（不超过aoi最大值）
    def update_aoi(self, l):
        success = np.random.random() < l.success_prob
        if success:
            l.state.aoi = 1
        else:
            l.state.aoi = min(l.aoi_max, l.state.aoi + 1)

    # 自定义——更新agent位置坐标
    def update_state(self, agent):
        noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
        agent.action.u += noise

        # 根据实际动作意义转换动作范围
        dir = (agent.action.u[0] + 1) * np.pi  # 方向范围 0-2pi
        vel = (agent.action.u[1] + 1) / 2 * agent.v_max  # 速度范围 0-vmax

        # 转换调度设备编号的动作范围 (-1,1)->(0,UE数量)
        # sch = int((len(self.landmarks) + 1) * ((agent.action.u[2] + 1) / 2))  # 考虑暂不调度任何设备的情况
        # if agent.action.u[2] == 1:
        #     sch = len(self.landmarks)
        sch = round((agent.action.u[2] + 1) / 2 * (len(self.landmarks) - 0))
        # print("agent.action.u[2]: ", agent.action.u[2], "sch: ", sch)

        # 分别更新agent的横纵坐标 u[0]是方向 u[1]是速度 u[2]是调度的设备编号
        agent.state.p_vel = vel
        agent.state.p_pos[0] += vel * self.t * math.cos(dir)
        agent.state.p_pos[1] += vel * self.t * math.sin(dir)
        agent.state.c[0] = sch  # 选择调度设备的编号 作为agent的通信状态
        # 更新UE的状态——如果被agent选择调度且在当前agent的覆盖范围内，则更新调度状态
        agent.state.c[1] = 0
        for i, l in enumerate(self.landmarks):
            if i == agent.state.c[0]:  # agent调度的是当前UE
                delta_pos = agent.state.p_pos[:2] - l.state.p_pos[:2]
                dis = np.sqrt(np.sum(np.square(delta_pos)))
                if dis <= agent.state.p_pos[2] * math.tan(math.radians(agent.angle)):  # 且在agent覆盖范围内
                    if l.state.is_connected == 0:  # 且未被调度过
                        l.state.is_connected = 1  # 更新该设备的调度状态为1 表示已被调度过
                        agent.state.c[1] = 1  # 更新无人机的调度状态为1 表示首次调度
                        if l.state.type == 1:  # 如果成功调度的是时延容忍的数据 则改变当前无人机的携带数据状态
                            agent.state.carry[i] = 1

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force
