import numpy as np
from grid import GridWorld,GridWorldV3,action2icon
from vi5 import visualStateValue,visualDelta

def transPolicy(policy):
    policy_new = policy.copy()
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            policy_new[i][j] = action2icon(policy[i][j])
    return policy_new

def policy_evaluation(grid, policy, V, theta=1e-4, max_iter=1000):
    """策略评估：计算当前策略下的价值函数"""
    for _ in range(max_iter):
        delta = 0
        new_V = V.copy()  # 创建 V 的副本
        for i in range(grid.rows):
            for j in range(grid.cols):
                state = (i, j)
                if state in grid.terminal_states:
                    continue  # 终止状态不更新
                #这个相当于π  get_next_state相当于模型P(s′∣s,a)=1
                action = policy[i, j]
                next_state = grid.get_next_state(state, action) # 使用状态转移模型
                #这个get_reward相当于模型 R(s,a,s′)=函数返回值
                reward = grid.get_reward(state, next_state, action)  # 使用奖励模型
                # 贝尔曼方程更新
                new_value = reward + grid.gamma * V[next_state]
                delta = max(delta, abs(new_value - V[state]))
                new_V[state] = new_value  # 更新副本 new_V
        V = new_V  # 一次迭代结束后，更新 V
        if delta < theta:
            break
    return V


def policy_improvement(grid, V):
    """策略改进：根据当前价值函数更新策略"""
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
                reward = grid.get_reward(state, next_state, action)
                value = reward + grid.gamma * V[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[i, j] = best_action[0].upper()  # 取动作首字母
    return policy


def policy_iteration(grid, theta=1e-4, max_iter=1000):
    """策略迭代算法"""
    # 初始化策略（随机策略）
    policy = np.random.choice(grid.actions, size=(grid.rows, grid.cols))
    # 初始化价值函数
    V = np.zeros((grid.rows, grid.cols))

    for iter  in range(max_iter):
        print(f"\n=== Iteration {iter} ===")
        # 策略评估
        V = policy_evaluation(grid, policy, V, theta)
        print("Value Function:\n", np.round(V, 2))
        # 策略改进
        new_policy = policy_improvement(grid, V)
        visualPolicy = transPolicy(new_policy)
        print("Policy:\n", visualPolicy)
        title = f"Iter={iter}"
        path = f"D:\\bak\github\\big_talk_blog\policy_iter_algo\imgs\grid_v3_{iter}.png"
        visualStateValue(V, visualPolicy, grid.terminal_states, grid.forbid_states, title=title, savePath=path)
        # 检查策略是否稳定
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V


if __name__ == "__main__":
    # 初始化网格世界，终点在(2,2)
    #terminal_states = {(3, 2)}
    #forbid_states = {}
    #grid = GridWorld(rows=5,cols=5,terminal_states=terminal_states, gamma=0.9)

    #带有随机奖励的网格
    #grid = GridWorldV2(terminal_states={(2, 2)}, gamma=0.9)

    # 带有禁止区域的网格世界
    terminal_states = {(3, 2)}
    forbid_states = {(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}
    grid = GridWorldV3(rows=5,cols=5,terminal_states=terminal_states, forbid_states=forbid_states,gamma=0.9)

    # 运行策略迭代算法
    optimal_policy, optimal_V = policy_iteration(grid, theta=1e-4)
    # 打印结果
    print("最优策略：\n", transPolicy(optimal_policy))
    print("最优价值函数（收敛后）：\n", np.round(optimal_V, 2))

    """
    # 提取最优策略
    optimal_policy = extract_policy(grid, optimal_V)
    print("最优策略：\n", optimal_policy)
    visualStateValue(optimal_V, optimal_policy, terminal_states, forbid_states)
    # 打印结果（保留两位小数）
    print("最优价值函数（收敛后）：\n", np.round(optimal_V, 2))

    # 打印误差
    visualDelta(deltaIter)
    """
