import numpy as np
from collections import defaultdict
from grid import GridWorld,GridWorldV2,GridWorldV3,action2icon
from vi import plot_epi

class MonteCarloAgent:
    def __init__(self, grid, gamma=0.99, epsilon=0.1):
        self.ep = 0.1  # 探索率
        self.env = grid
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(self.env.actions)))
        self.N = defaultdict(lambda: np.zeros(len(self.env.actions)))

    def generate_episode(self, max_steps):
        #生成一个episode，使用episode-greedy策略
        state = self.env.reset() # 仅通过环境交互获取初始状态
        episode = []
        done = False
        steps = 0
        num_actions = len(self.env.actions)

        while not done and steps < max_steps:
            #ep概率随机选择动作 否则选择最优动作
            if np.random.rand() < self.ep:
                action_idx = np.random.choice(num_actions)
            else:
                action_idx = np.argmax(self.Q[state])

            #action_idx = np.argmax(self.Q[state])
            action = self.env.actions[action_idx]  # 基于策略选择动作
            next_state, reward, done = self.env.step(state, action)  # 通过环境交互获取下一状态和奖励
            episode.append((state, action_idx, reward))
            state = next_state
            steps += 1
        return episode  # 直接使用轨迹数据，不涉及环境模型

    def train(self,num_episodes, max_steps):
        for epi_iter in range(num_episodes):
            print("episode ",epi_iter)
            """
            智能体不需要知道环境如何生成next_state或reward，只需通过交互获取数据。
            例如，在网格世界中，智能体不知道向右移动是否会成功到达(0, 1)，它只能通过尝试动作并观察结果。
            """
            #产生一条episode
            episode = self.generate_episode(max_steps)
            print(episode)
            #打印轨迹
            #self.env.print_episode_path(episode)
            #if epi_iter > 300:
            #    plot_epi(episode, (self.env.rows, self.env.cols))
            G = 0
            visited = set()

            # 根据episode获得的状态，动作，奖励更新Q
            for t in reversed(range(len(episode))):
                state, action_idx, reward = episode[t]
                G = self.gamma * G + reward

                if (state, action_idx) not in visited:
                    visited.add((state, action_idx))
                    self.N[state][action_idx] += 1 #记录出现次数
                    #计算平均值-省存储的方法
                    self.Q[state][action_idx] += (G - self.Q[state][action_idx]) / self.N[state][action_idx]
                    #print(self.Q)
    def get_policy(self):
        """从Q表提取最优策略"""
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy



# 训练与测试
if __name__ == "__main__":
    # 创建环境与智能体
    #grid_world = GridWorld(terminal_states={(2, 2)}, gamma=0.9)
    grid_world = GridWorldV3(rows=5, cols=5, terminal_states={(3, 2)},
                       forbid_states={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}, gamma=0.9)

    agent = MonteCarloAgent(grid_world, gamma=0.9, epsilon=0.5)

    # 执行训练
    print("开始训练...")
    num_episodes = 2000  # 训练episode数量
    num_steps = 500  # 单次episode最大步数
    agent.train(num_episodes,num_steps)
    print("训练完成！")

    # 获取并显示最优策略
    policy = agent.get_policy()
    grid_world.render_policy(policy)

    # 测试路径演示
    state = grid_world.reset()
    done = False
    path = [state]
    steps = 0
    max_steps = 1000
    while not done:
        action_idx = np.argmax(agent.Q[state])
        action = grid_world.actions[action_idx]
        next_state, _, done = grid_world.step(state, action)
        path.append(next_state)
        state = next_state
        steps += 1
        #print(path)

    print("\n最优路径演示：")
    print(" → ".join(map(str, path)))