import matplotlib.pyplot as plt
import numpy as np

def visualStateValue(state_values, policy, terminal_states = None, forbid_states = None, title = "", savePath = None):
    """
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
"""
    # 网格大小
    rows, cols = state_values.shape
    print("debug",rows, cols)
    # 创建一个图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制热力图
    #cax = ax.matshow(state_values, cmap='viridis')
    #fig.colorbar(cax)
    # 绘制背景颜色为蓝色
    color = np.full((rows, cols), 0.5)  # 使用灰色级别表示颜色，1为白色，0为黑色。这里用0.8代表较浅的蓝色效果。
    ax.imshow(color, cmap='Blues', aspect='equal')
    # 将 termine state 的网格设置为黄色
    if terminal_states:
        for i,j in terminal_states:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="yellow", edgecolor="black", linewidth=1)
            ax.add_patch(rect)
    if forbid_states:
        for i,j in forbid_states:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", edgecolor="black", linewidth=1)
            ax.add_patch(rect)
    # 在每个格子上添加值和策略方向
    for i in range(rows):
        for j in range(cols):
            # 添加状态值
            ax.text(j, i, f"{state_values[i, j]:.3f}", va='center', ha='center', color='black', fontsize=12)

            # 添加策略方向
            #if policy[i, j] != 'T':  # 如果不是终止状态
                #print(i,j)
                #ax.text(j, i + 0.3, policy[i, j], va='center', ha='center', color='red', fontsize=14, fontweight='bold')
            ax.text(j, i+0.3, policy[i, j], va='center', ha='center', color='red', fontsize=14, fontweight='bold')

    # 设置网格线
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    # 设置坐标轴
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xticklabels(np.arange(1, cols + 1))  # 可选：设置列标签为1, 2, 3...
    ax.set_yticklabels(np.arange(1, rows + 1))  # 可选：设置行标签为1, 2, 3...

    # 反转y轴以符合矩阵索引习惯
    #ax.invert_yaxis()
    # 将x轴的标签放置在顶部
    ax.xaxis.tick_top()  # 将x轴刻度移到顶部
    ax.xaxis.set_label_position("top")  # 将x轴标签移到顶部
    if title:
        ax.set_xlabel(title, fontsize=14)  # 设置x轴标签
    # 添加标题
    plt.title("State Values and Policy Visualization")
    #保存图片
    if savePath:
        plt.savefig(savePath)
    else:
        # 显示图形
        plt.show()
def visualGrid(rows,cols, terminal_states = None, forbid_states = None):

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制背景颜色为蓝色
    color = np.full((rows, cols), 0.5)  # 使用灰色级别表示颜色，1为白色，0为黑色。这里用0.8代表较浅的蓝色效果。
    ax.imshow(color, cmap='Blues', aspect='equal')
    # 将 termine state 的网格设置为黄色
    if terminal_states:
        for i,j in terminal_states:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="yellow", edgecolor="black", linewidth=1)
            ax.add_patch(rect)
    if forbid_states:
        for i,j in forbid_states:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", edgecolor="black", linewidth=1)
            ax.add_patch(rect)

    # 设置网格线
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    # 设置坐标轴
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xticklabels(np.arange(1, cols + 1))  # 可选：设置列标签为1, 2, 3...
    ax.set_yticklabels(np.arange(1, rows + 1))  # 可选：设置行标签为1, 2, 3...

    # 将x轴的标签放置在顶部
    ax.xaxis.tick_top()  # 将x轴刻度移到顶部
    ax.xaxis.set_label_position("top")  # 将x轴标签移到顶部
    #ax.set_xlabel("X Axis Label", fontsize=14)  # 设置x轴标签
    # 添加标题
    plt.title("Grid Visualization")

    # 显示图形
    plt.show()
if __name__ == "__main__":
    """
    policy = np.array([['↓', '↓' ,'↓' ,'↓' ,'↓'],
     ['↓', '↓' ,'↓' ,'↓', '↓'],
     ['↓' ,'↓', '↓', '↓', '↓'],
     ['→', '→', 'T' ,'←', '←'],
     ['↑', '↑' ,'↑', '↑', '↑']])
    state = np.array([[ 6.56,  7.29,  8.1,   7.29,  6.56],
 [ 7.29,  8.1 ,  9.  ,  8.1 ,  7.29],
 [ 8.1,   9.  , 10.  ,  9.  ,  8.1 ],
 [ 9. ,  10. ,  10.  , 10.  ,  9.  ],
 [ 8.1 ,  9. ,  10. ,   9.  ,  8.1 ]])
    visualStateValue(state, policy, terminal_states={(3, 2)}, forbid_states = {(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)})
    """
    visualGrid(5,5, terminal_states={(3, 2)}, forbid_states = {(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)})