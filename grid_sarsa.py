import random
import numpy as np
from collections import defaultdict


class GridWorld:
    def __init__(self, rows=4, cols=4, start=(0, 0), end=(3, 3)):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end
        self.actions = [0, 1, 2, 3]  # 上、下、左、右
        self.action_names = ['↑', '↓', '←', '→']

    def reset(self):
        return self.start

    def step(self, state, action):
        x, y = state
        if action == 0:  # 上
            x = max(x - 1, 0)
        elif action == 1:  # 下
            x = min(x + 1, self.rows - 1)
        elif action == 2:  # 左
            y = max(y - 1, 0)
        elif action == 3:  # 右
            y = min(y + 1, self.cols - 1)
        next_state = (x, y)
        done = (next_state == self.end)
        reward = 0 if done else -1
        return next_state, reward, done

    def epsilon_greedy(self, Q, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax([Q[(state, a)] for a in self.actions])
def sarsa(env, alpha=0.1, gamma=1.0, epsilon=0.1, episodes=10000):
    Q = defaultdict(float)

    for _ in range(episodes):
        state = env.reset()
        action = env.epsilon_greedy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = env.step(state, action)
            next_action = env.epsilon_greedy(Q, next_state, epsilon)

            # SARSA更新公式
            td_target = reward + gamma * Q[(next_state, next_action)]
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state, action = next_state, next_action

    return Q


def q_learning(env, alpha=0.1, gamma=1.0, epsilon=0.1, episodes=10000):
    Q = defaultdict(float)

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = env.epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)

            # Q-learning更新公式
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            td_target = reward + gamma * max_next_q
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])

            state = next_state

    return Q


def print_q_table(env, Q):
    print("Q-table:")
    for i in range(env.rows):
        for j in range(env.cols):
            state = (i, j)
            if state == env.end:
                print(f"State {state} (Terminal)")
                continue
            q_values = [Q[(state, a)] for a in env.actions]
            arrows = [f"{env.action_names[a]}:{q:.2f}" for a, q in zip(env.actions, q_values)]
            print(f"State {state}: {', '.join(arrows)}")
        print("+" + "-" * 50 + "+")



def compare_results(env, sarsa_q, qlearn_q):
    print("\nSARSA策略：")
    print_q_table(env, sarsa_q)

    print("\nQ-learning策略：")
    print_q_table(env, qlearn_q)


if __name__ == "__main__":
    env = GridWorld()

    # 训练参数
    params = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "episodes": 20000
    }

    # 训练算法
    sarsa_q = sarsa(env, **params)
    qlearn_q = q_learning(env, **params)

    # 结果对比
    compare_results(env, sarsa_q, qlearn_q)