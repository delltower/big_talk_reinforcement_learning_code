import matplotlib.pyplot as plt
import numpy as np

# 当前状态值 (state value)
state_values = np.array([
    [0.729, 0.81, 0.9],
    [0.81, 0.9, 1.0],
    [0.9, 1.0, 0.0]
])

# 当前策略 (policy)
policy = np.array([
    ['↓', '↓', '↓'],
    ['↓', '↓', '↓'],
    ['→', '→', 'T']
])

# 网格大小
rows, cols = state_values.shape

# 创建一个图形
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制热力图
cax = ax.matshow(state_values, cmap='viridis')
fig.colorbar(cax)

# 在每个格子上添加值和策略方向
for i in range(rows):
    for j in range(cols):
        # 添加状态值
        ax.text(j, i, f"{state_values[i, j]:.3f}", va='center', ha='center', color='white', fontsize=12)

        # 添加策略方向
        if policy[i, j] != 'T':  # 如果不是终止状态
            ax.text(j, i + 0.3, policy[i, j], va='center', ha='center', color='red', fontsize=14, fontweight='bold')

# 设置坐标轴
ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))
ax.set_xticklabels(np.arange(1, cols + 1))  # 可选：设置列标签为1, 2, 3...
ax.set_yticklabels(np.arange(1, rows + 1))  # 可选：设置行标签为1, 2, 3...

# 反转y轴以符合矩阵索引习惯
ax.invert_yaxis()

# 添加标题
plt.title("State Values and Policy Visualization")

# 显示图形
plt.show()