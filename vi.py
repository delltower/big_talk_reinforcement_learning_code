import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_epi(episode, grid_size):

    # 初始化访问计数器
    visit_count = {}
    """
    for state, _, _ in episode:
        if state not in visit_count:
            visit_count[state] = 0
    """
    for state in episode:
        if state not in visit_count:
            visit_count[state] = 0
    # 轨迹列表，用于存储每一步的位置
    trajectory = []
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, grid_size[1]-0.5)
        ax.set_ylim(grid_size[0]-0.5, -0.5)
        ax.set_xticks(range(grid_size[1]))
        ax.set_yticks(range(grid_size[0]))
        ax.grid(True)

        # 更新访问计数器
        #state, action, reward = episode[frame]
        state= episode[frame]
        visit_count[state] += 1
        trajectory.append(state)

        # 绘制访问次数
        for (x, y), count in visit_count.items():
            ax.text(y + 0.5, x + 0.9, str(count), ha='center', va='center', color='blue') # 显示在单元格上方

        # 绘制轨迹
        if len(trajectory) > 1:
            xs, ys = zip(*[(pos[1] + 0.5, pos[0] + 0.5) for pos in trajectory])
            ax.plot(xs, ys, 'b-', linewidth=2)  # 使用红线连接点表示轨迹
        # 标记当前步骤的位置
        current_position = episode[frame][0]
        # 调整位置以确保标记位于单元格中央
        ax.plot(current_position[1] + 0.5, current_position[0] + 0.5, 'r*', markersize=15) # 当前位置标记为红色星号，并适当调整位置和大小

        ax.set_title(f"Step: {frame}, Position: {state}, Action: {action}, Reward: {reward}")
        # 如果是最后一帧，设置回调函数在动画结束后关闭图形
        if frame == len(episode) - 1:
            ani.event_source.stop()
            plt.pause(1)
            plt.close(fig)

    #fig, ax = plt.subplots()
    # 设置图形窗口大小，这里设为宽6英寸，高4英寸
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ani = animation.FuncAnimation(fig, update, frames=len(episode), repeat=False)
    plt.show(block=True)  # 确保图形窗口保持打开直到动画结束

if __name__ == "__main__":
    num_episodes = 3
    for epi_iter in range(num_episodes):
        print("Episode", epi_iter)
        #episode = [((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 2, -1),
        #           ((1, 1), 0, -1)]  # 示例数据
        episode = [(0, 0),(0, 0), (0, 1), (0, 1), (0, 1),(1, 1)]
        plot_epi(episode, (3, 4))  # 假设网格大小为3x4
        time.sleep(2)  # 每次调用之间增加一秒的延迟
    """
    # 示例episode数据
    episode = [((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1),
               ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1),
               ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 0, 0),
               ((0, 0), 4, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0),
               ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0),
               ((0, 1), 2, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1),
               ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1),
               ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 4, -1), ((1, 2), 1, 0), ((0, 2), 3, 0),
               ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 1, -1), ((0, 1), 0, 0), ((0, 1), 4, 0), ((0, 2), 3, 0),
               ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0),
               ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 0, 0), ((0, 1), 2, -1), ((1, 1), 0, -1), ((1, 1), 0, -1),
               ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1),
               ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1), ((1, 1), 0, -1),
               ((1, 1), 0, -1), ((1, 1), 3, 0), ((1, 0), 1, 0), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1),
               ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1),
               ((0, 0), 0, 0), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1),
               ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1), ((0, 0), 3, -1)]

    # 定义网格大小
    grid_size = (3, 4)  # 根据你的数据调整尺寸
    plot_epi(episode, grid_size)
    """