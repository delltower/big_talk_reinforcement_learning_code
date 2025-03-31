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

#绘制delta和iter的曲线
def visualDelta(deltaIter):
    # 创建一个图形
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot([iter for iter, delta in deltaIter],[delta for iter, delta in deltaIter])
    plt.title("Delta with Iter")
    plt.xlabel("Iter")
    plt.ylabel("delta value")
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
    #绘制网格
    #visualGrid(5,5, terminal_states={(3, 2)}, forbid_states = {(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)})

    #绘制误差曲线
    deltaIter = [(0, 1.0), (1, 0.9), (2, 0.81), (3, 0.7290000000000001), (4, 0.6561000000000003), (5, 0.5904900000000004), (6, 0.5314410000000005), (7, 0.4782969000000006), (8, 0.43046721000000066), (9, 0.3874204890000006), (10, 0.3486784401000007), (11, 0.3138105960900006), (12, 0.282429536481001), (13, 0.25418658283290085), (14, 0.2287679245496106), (15, 0.20589113209465015), (16, 0.18530201888518505), (17, 0.16677181699666654), (18, 0.15009463529700007), (19, 0.13508517176730006), (20, 0.12157665459057032), (21, 0.10941898913151338), (22, 0.09847709021836204), (23, 0.08862938119652597), (24, 0.07976644307687408), (25, 0.07178979876918667), (26, 0.06461081889226783), (27, 0.05814973700304105), (28, 0.05233476330273712), (29, 0.04710128697246363), (30, 0.04239115827521722), (31, 0.03815204244769621), (32, 0.03433683820292721), (33, 0.030903154382634135), (34, 0.02781283894437081), (35, 0.025031555049934262), (36, 0.02252839954494057), (37, 0.02027555959044669), (38, 0.018248003631402554), (39, 0.0164232032682623), (40, 0.01478088294143598), (41, 0.013302794647292338), (42, 0.01197251518256337), (43, 0.010775263664307033), (44, 0.009697737297876152), (45, 0.008727963568088803), (46, 0.0078551672112801), (47, 0.0070696504901519575), (48, 0.006362685441136939), (49, 0.005726416897023245), (50, 0.005153775207320965), (51, 0.004638397686588913), (52, 0.0041745579179304215), (53, 0.003757102126137557), (54, 0.0033813919135239345), (55, 0.003043252722171541), (56, 0.002738927449954076), (57, 0.002465034704958846), (58, 0.0022185312344635832), (59, 0.001996678111017225), (60, 0.0017970102999154136), (61, 0.0016173092699238723), (62, 0.0014555783429317515), (63, 0.0013100205086384875), (64, 0.0011790184577753493), (65, 0.0010611166119982585), (66, 0.000955004950798255), (67, 0.0008595044557182518), (68, 0.0007735540101467819), (69, 0.0006961986091322814), (70, 0.0006265787482195861), (71, 0.0005639208733976275), (72, 0.0005075287860578648), (73, 0.00045677590745185626), (74, 0.000411098316706493), (75, 0.00036998848503611015), (76, 0.0003329896365329432), (77, 0.00029969067287982654), (78, 0.00026972160559157743), (79, 0.00024274944503233087), (80, 0.00021847450052892015), (81, 0.00019662705047629458), (82, 0.0001769643454290204), (83, 0.0001592679108863848), (84, 0.00014334111979774633), (85, 0.00012900700781814933), (86, 0.00011610630703628999), (87, 0.0001044956763331939), (88, 9.404610870067387e-05)]
    visualDelta(deltaIter)