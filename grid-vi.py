import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from grid import GridWorld, extract_policy


def value_iteration_visualize(grid, theta=1e-4, max_iter=1000):
    # 初始化价值函数和图形
    V = np.zeros((grid.rows, grid.cols))
    fig, ax = plt.subplots()
    im = ax.imshow(V, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im)
    ax.set_xticks(np.arange(grid.cols))
    ax.set_yticks(np.arange(grid.rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    def update(frame):
        nonlocal V
        new_V = np.copy(V)
        delta = 0
        for i in range(grid.rows):
            for j in range(grid.cols):
                state = (i, j)
                if state in grid.terminal_states:
                    continue
                max_value = -np.inf
                for action in grid.actions:
                    next_state = grid.get_next_state(state, action)
                    reward = grid.get_reward(state, next_state)
                    value = reward + grid.gamma * V[next_state]
                    max_value = max(max_value, value)
                new_V[i, j] = max_value
                delta = max(delta, abs(new_V[i, j] - V[i, j]))
                print("debug delta",delta)
        V = np.copy(new_V)
        im.set_data(V)
        ax.set_title(f'Iteration {frame}, Delta: {delta:.4f}')

        # 如果 delta < theta，停止动画
        if delta < theta:
            ani.event_source.stop()

        return [im]

    ani = FuncAnimation(fig, update, frames=max_iter, interval=500, blit=True)
    plt.show()
    return V

# 初始化网格世界，终点在(2,2)
grid = GridWorld(terminal_states={(2, 2)}, gamma=0.9)

# 运行值迭代算法
optimal_V = value_iteration_visualize(grid, theta=1e-4, max_iter= 30)

# 提取最优策略
optimal_policy = extract_policy(grid, optimal_V)
print("最优策略：\n", optimal_policy)

# 打印结果（保留两位小数）
print("最优价值函数（收敛后）：\n", np.round(optimal_V, 2))