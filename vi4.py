import numpy as np
import matplotlib.pyplot as plt

# 创建一个5x5的矩阵，用于表示网格
data = np.zeros((5, 5))

# 设置中间位置的值
data[2, 2] = 1  # 中心点
data[1:4, 1:4] = 0.5  # 周围区域

# 创建颜色映射
cmap = plt.cm.get_cmap('viridis')
cmap.set_bad(color='white')  # 设置无效值的颜色为白色

# 绘制网格
plt.imshow(data, cmap=cmap, interpolation='nearest', aspect='equal')

# 添加网格线
plt.grid(True, which='both', linestyle='-', linewidth=1)

# 添加刻度标签
plt.xticks(np.arange(5) + 0.5, ['1', '2', '3', '4', '5'])
plt.yticks(np.arange(5) + 0.5, ['1', '2', '3', '4', '5'])

# 调整刻度标签的位置
plt.gca().set_xticks(np.arange(5), minor=True)
plt.gca().set_yticks(np.arange(5), minor=True)
plt.tick_params(which='minor', length=0)

# 显示图形
plt.show()