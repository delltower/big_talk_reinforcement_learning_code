import matplotlib.pyplot as plt
import numpy as np

def visualStateValue(state_values, policy):
    """
    可视化状态值和策略方向
    :param state_values: 状态值矩阵 (numpy array)
    :param policy: 策略矩阵 (numpy array)
    """
    # 网格大小
    rows, cols = state_values.shape
    print("Debug: Grid size:", rows, cols)

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制背景颜色为蓝色
    color = np.full((rows, cols), 0.8)  # 使用灰色级别表示颜色，1为白色，0为黑色。这里用0.8代表较浅的蓝色效果。
    ax.imshow(color, cmap='Blues', aspect='equal')

    # 在每个格子上添加值和策略方向
    for i in range(rows):
        for j in range(cols):
            # 添加状态值
            ax.text(j, i, f"{state_values[i, j]:.2f}", va='center', ha='center', color='white', fontsize=12)

            # 添加策略方向
            if policy[i, j] != 'T':  # 如果不是终止状态
                ax.text(j, i + 0.3, policy[i, j], va='center', ha='center', color='red', fontsize=14, fontweight='bold')

    # 设置网格线
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # 设置坐标轴标签
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(1, cols + 1))  # 列标签从1开始
    ax.set_yticklabels(np.arange(1, rows + 1))  # 行标签从1开始

    # 反转y轴以符合矩阵索引习惯
    ax.invert_yaxis()

    # 添加标题
    plt.title("State Values and Policy Visualization", fontsize=16)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 示例策略和状态值
    policy = np.array([
        ['↓', '↓', '↓', '↓', '↓'],
        ['↓', '↓', '↓', '↓', '↓'],
        ['↓', '↓', '↓', '↓', '↓'],
        ['→', '→', 'T', '←', '←'],
        ['↑', '↑', '↑', '↑', '↑']
    ])
    state = np.array([
        [6.56, 7.29, 8.1, 7.29, 6.56],
        [7.29, 8.1, 9.0, 8.1, 7.29],
        [8.1, 9.0, 10.0, 9.0, 8.1],
        [9.0, 10.0, 10.0, 10.0, 9.0],
        [8.1, 9.0, 10.0, 9.0, 8.1]
    ])
    visualStateValue(state, policy)