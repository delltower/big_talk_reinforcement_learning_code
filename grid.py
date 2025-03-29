import numpy as np
import random

class GridWorldBase:
    def __init__(self, rows = 3, cols = 3, terminal_states = None, forbid_states = None, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.terminal_states = terminal_states if terminal_states else {(rows-1, cols-1)}
        self.actions = ['stay','up','down','left','right']
        self.action_arrows = ['0','↑', '↓', '←', '→']  # 动作对应的符号
        self.gamma = gamma
        self.forbid_states = forbid_states if forbid_states else set()

    def reset(self):
        return (0, 0)  # 起始位置在左上角
    def get_next_state(self, state, action):
        i, j = state
        if action == 'up' or action == 'U':
            next_i = max(i - 1, 0)
            next_j = j
        elif action == 'down' or action == 'D':
            next_i = min(i + 1, self.rows - 1)
            next_j = j
        elif action == 'left' or action == 'L':
            next_i = i
            next_j = max(j - 1, 0)
        elif action == 'right' or action == 'R':
            next_i = i
            next_j = min(j + 1, self.cols - 1)
        elif action == 'stay' or action == 'S' or action == 'T':
            next_i = i
            next_j = j
        return (next_i, next_j)

    def get_reward(self, state, next_state):
        #奖励函数
        pass

    def step(self, state, action):
        # 获取下个状态 奖励 是否结束
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, next_state, action)
        #print("debug step", state, action, next_state)
        done = next_state in self.terminal_states
        return next_state, reward, done

    def print_episode_path(self, episode):
        """
        在网格中打印 episode 的路径
        :param episode: episode 数据，格式为 [(state, action_idx, reward), ...]
        :param grid_world: GridWorld 环境对象
        """
        # 初始化网格
        grid = np.full((self.rows, self.cols), ' ', dtype=object)

        # 遍历 episode，填充网格
        for step in episode:
            state, action_idx, _ = step
            i, j = state
            grid[i][j] = self.action_arrows[action_idx]  # 将动作映射为箭头符号

        # 打印网格
        print("Episode 路径可视化：")
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(f"{grid[i][j]:^3}")  # 每个格子占 3 个字符宽度
            print(" ".join(row))
    def get_policy(self, Q):
        """从Q表提取最优策略"""
        policy = {}
        for state in Q:
            policy[state] = np.argmax(Q[state])
        return policy

    def render_policy(self, policy):
        """可视化显示策略"""
        grid = np.full((self.rows, self.cols), '', dtype=object)
        for (x, y), action_idx in policy.items():
            grid[x][y] = self.action_arrows[action_idx]
        print("最优策略矩阵：")
        for row in grid:
            print(" ".join(f"{arrow:^3}" for arrow in row))

class GridWorld(GridWorldBase):
    #有边界
    def __init__(self, rows=3, cols=3, terminal_states=None, gamma=0.9):
        super().__init__(rows, cols, terminal_states, gamma)

    def get_reward(self, state, next_state, action):
        #奖励函数 边界惩罚
        i, j = state

        # 检查是否触碰边界
        is_boundary_hit = False
        if action == 'up' and i == 0:  # 向上触碰上边界
            is_boundary_hit = True
        elif action == 'down' and i == self.rows - 1:  # 向下触碰下边界
            is_boundary_hit = True
        elif action == 'left' and j == 0:  # 向左触碰左边界
            is_boundary_hit = True
        elif action == 'right' and j == self.cols - 1:  # 向右触碰右边界
            is_boundary_hit = True

        # 计算奖励
        if next_state in self.terminal_states:
            return 1  # 到达终止状态
        elif is_boundary_hit:
            return -1  # 触碰边界惩罚
        else:
            return 0  # 其他情况

class GridWorldV3(GridWorldBase):
    #增加禁止区域和边界惩罚和随机性奖励
    def __init__(self, rows=3, cols=3, terminal_states=None, forbid_states=None, gamma=0.9, randomness=0.1):
        super().__init__(rows, cols, terminal_states, forbid_states, gamma)
        self.randomness = randomness  # 随机性概率

    def get_reward(self, state, next_state, action):
        # 奖励函数
        i,j = state

        # 检查是否触碰边界
        is_boundary_hit = False
        if action == 'up' and i == 0:  # 向上触碰上边界
            is_boundary_hit = True
        elif action == 'down' and i == self.rows - 1:  # 向下触碰下边界
            is_boundary_hit = True
        elif action == 'left' and j == 0:  # 向左触碰左边界
            is_boundary_hit = True
        elif action == 'right' and j == self.cols - 1:  # 向右触碰右边界
            is_boundary_hit = True

        if next_state in self.forbid_states:
            reward = -10
        elif is_boundary_hit:
            reward = -5
        elif next_state in self.terminal_states:
            reward = 1
        else:
            reward = 0
        return reward

class GridWorldV2(GridWorldBase):
    #增加随机奖励
    def __init__(self, rows=3, cols=3, terminal_states=None, gamma=0.9, randomness=0.1):
        super().__init__(rows, cols, terminal_states, gamma=gamma)
        self.randomness = randomness

    def get_reward(self, state, next_state):
        # 奖励函数中加入随机性
        base_reward = 1 if next_state in self.terminal_states else 0
        if random.random() < self.randomness:
            return base_reward + random.uniform(-0.5, 0.5)  # 随机波动
        return base_reward

class GridWorldV4(GridWorld):
    #增加e-greedy策略
    def __init__(self, rows=3, cols=3, terminal_states=None, gamma=0.9):
        super().__init__(rows, cols, terminal_states, gamma)

    def epsilon_greedy(self, Q, state, ep):
        if np.random.rand() < ep:
            action_idx = np.random.choice(len(self.actions))
            return  action_idx
        else:
            q_values = [Q[state][a] for a in range(len(self.actions))]
            max_q = max(q_values)
            max_action_idx = [a for a in range(len(self.actions)) if Q[state][a] == max_q]
            return np.random.choice(max_action_idx)

    def get_reward(self, state, next_state, action):
        # 奖励函数
        if next_state in self.terminal_states:
            reward = 0
        else:
            reward = 1
        return reward

def action2icon(action):
    #动作转换成icon
    if action == 'up' or action == 'U':
        return '↑'
    elif action == 'down' or action == 'D':
        return '↓'
    elif action == 'left' or action == 'L':
        return '←'
    elif action == 'right' or action == 'R':
        return '→'
    elif action == 'stay' or action == 'S' or action == 'T':
        return '0'
    return ''
def extract_policy(grid, V):
    """根据最优价值函数提取最优策略"""
    policy = np.empty((grid.rows, grid.cols), dtype=str)
    for i in range(grid.rows):
        for j in range(grid.cols):
            state = (i, j)
            if state in grid.terminal_states:
                policy[i, j] = 'T'  # 终止状态
                continue
            max_value = -np.inf
            best_action = None
            for action in grid.actions:
                next_state = grid.get_next_state(state, action)
                value = grid.get_reward(state, next_state, action) + grid.gamma * V[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[i, j] = action2icon(best_action[0].upper())  # 取动作首字母
    return policy
