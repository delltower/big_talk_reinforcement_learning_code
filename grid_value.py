import numpy as np
import random
from grid import GridWorldV3,extract_policy,GridWorld
from vi5 import visualStateValue

def value_iteration(grid, theta=1e-4, max_iter = 1000):
    #值迭代算法
    #初始化state 函数
    V = np.zeros((grid.rows, grid.cols))

    for Iter in range(max_iter):
        delta = 0
        new_V = np.copy(V)
        print("---------Iter------------", Iter)
        for i in range(grid.rows):
            for j in range(grid.cols):
                state = (i,j)
                if state in grid.terminal_states:
                    ##终止状态不更新 动作为原地不动
                    new_V[i,j] = 1.0 + grid.gamma*V[i,j]
                    continue
                #计算所有可能动作的值函数
                max_value = -np.inf
                for action in grid.actions:
                    next_state = grid.get_next_state(state, action)
                    reward = grid.get_reward(state, next_state, action)
                    #贝尔曼最优方程更新
                    value = reward + grid.gamma * V[next_state]
                    if value > max_value:
                        max_value = value
                new_V[i,j] = max_value
                delta = max(delta, abs(new_V[i,j]-V[i,j]))

        V = np.copy(new_V)
        print("当前的state value：\n",V)
        iter_policy = extract_policy(grid, V)
        print("当前策略：\n", iter_policy)
        print("debug delta", f"{delta:.20f}")
        title = f"Iter={Iter}, delta={delta:.20f}"
        path=f"./grid_value/{Iter}_value.png"
        visualStateValue(V, iter_policy, grid.terminal_states, {},title = title, savePath=path)
        if delta < theta:
            break
    return V

if __name__ == "__main__":
    # 初始化网格世界，终点在(2,2)
    terminal_states = {(3, 2)}
    forbid_states = {}
    grid = GridWorld(rows=5,cols=5,terminal_states=terminal_states, gamma=0.9)
    #带有随机奖励的网格
    #grid = GridWorldV2(terminal_states={(2, 2)}, gamma=0.9)
    #terminal_states = {(3, 2)}
    #forbid_states = {(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)}
    #带有禁止区域的网格世界
    #grid = GridWorldV3(rows=5,cols=5,terminal_states=terminal_states, forbid_states=forbid_states,gamma=0.9)

    # 运行值迭代算法
    optimal_V = value_iteration(grid, theta=1e-4, max_iter = 500)

    # 提取最优策略
    optimal_policy = extract_policy(grid, optimal_V)
    print("最优策略：\n", optimal_policy)
    visualStateValue(optimal_V, optimal_policy, terminal_states, forbid_states)
    # 打印结果（保留两位小数）
    print("最优价值函数（收敛后）：\n", np.round(optimal_V, 2))