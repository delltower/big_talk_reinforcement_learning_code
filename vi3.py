import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.arange(10)
y = np.random.rand(10)

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(x, y, marker='o', label='Data')

# 设置 X 轴刻度位置和标签
xticks_positions = np.arange(-0.5, 10, 1)  # 刻度位置为每个整数之间
xticks_labels = [str(i) for i in range(10)]  # 刻度标签为整数
plt.xticks(ticks=xticks_positions, labels=xticks_labels)
plt.xlim(-0.5, 9.5)  # 调整 X 轴范围

# 设置 Y 轴刻度位置和标签
yticks_positions = np.arange(-0.5, 1.1, 0.1)  # 刻度位置为每个 0.1 之间
yticks_labels = [f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)]  # 刻度标签为小数
plt.yticks(ticks=yticks_positions, labels=yticks_labels)
plt.ylim(-0.05, 1.05)  # 调整 Y 轴范围

# 启用网格并设置样式
plt.grid(which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')

# 添加标题和标签
plt.title('Grid with Labels in the Middle (Both Axes)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图例
plt.legend()

# 显示图形
plt.show()