import random
from collections import defaultdict
from grid import GridWorld,extract_policy,GridWorldV3
grid_world = GridWorldV3(rows=5, cols=5, terminal_states={(3, 2)},
                       forbid_states={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}, gamma=0.9)
def td0_demo():
    # 初始化环境参数
    env = grid_world
    alpha = 0.1  # 学习率
    gamma = 1.0  # 折扣因子
    num_episodes = 10000  # 训练幕数
    max_steps = 100  # 每幕最大步数

    # 初始化值函数
    V = defaultdict(float)

    # TD(0)算法
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # 随机选择动作
            action = random.choice(env.actions)
            next_state, reward, done = env.step(state, action)

            # TD(0)更新
            td_target = reward + gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])

            state = next_state
            steps += 1

    # 打印结果
    print("网格世界状态值函数：")
    print("+" + "-" * 35 + "+")
    for i in range(env.rows):
        print("|", end="")
        for j in range(env.cols):
            state = (i, j)
            if state in env.terminal_states:
                print(f"{'终点':^7}", end=" |")
            else:
                print(f"{V[state]:7.2f}", end=" |")
        print("\n+" + "-" * 35 + "+")
    # 提取最优策略
    optimal_policy = extract_policy(env, V)
    print("最优策略：\n", optimal_policy)

if __name__ == "__main__":
    td0_demo()