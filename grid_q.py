import random
import numpy as np
from collections import defaultdict
from grid import GridWorldV4
from vi2 import plot_epi, plot_epi_num, plot_delta_curve
def countReward(Q):
    return sum(q for state in Q for q in Q[state])  # 更简洁的写法


def compute_v(Q, state, epsilon):

    actions = [i for i in range(len(Q[state]))]
    q_values = Q[state]
    max_q = max(q_values)
    optimal_actions = []
    for i in range(len(Q[state])):
        if Q[state][i] == max_q:
            optimal_actions.append(i)
    k = len(optimal_actions)
    n = len(actions)

    total_v = 0.0
    for i in range(len(Q[state])):
        if max_q == q_values[i]:
            prob = (1 - epsilon) / k + epsilon / n
        else:
            prob = epsilon / n
        total_v += prob * q_values[i]
    return total_v

def sarsa(env, alpha=0.1, gamma=1.0, epsilon=0.1, episodes=10, max_steps = 10):
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    path = []
    eReward = []

    for iter_idx in range(episodes):
        delta = 0
        state = env.reset()
        action_idx = env.epsilon_greedy(Q, state, epsilon)
        #print("debug_init_action", action_idx, iter_idx)
        done = False
        steps = 0

        while not done and steps < max_steps:
            #根据策略生成轨迹的下一个状态
            next_state, reward, done = env.step(state, env.actions[action_idx])
            #根据策略生成下一个状态的action
            next_action_idx = env.epsilon_greedy(Q, next_state, epsilon)
            #print("debug node",steps, next_state, reward, done, next_action_idx)
            # SARSA更新公式
            td_target = reward + gamma * Q[next_state][next_action_idx]
            td_error = td_target - Q[state][action_idx]
            #print("debug value",td_target, td_error, reward, gamma, alpha)

            oldStateValue = compute_v(Q, state, epsilon)
            Q[state][action_idx] += alpha * td_error
            newStateValue = compute_v(Q, state, epsilon)
            delta = max(delta, abs(newStateValue-oldStateValue))
            state, action_idx = next_state, next_action_idx
            steps += 1

            path.append(state)
            """
            if len(path) == 0:
                path.append(state)
            elif state != path[-1]:
                path.append(state)
            else:
                continue
            """
        #print("debug sarsa ", iter_idx,  path)
        #for s in Q:
        #    print(s,Q[s])
        eReward.append((iter_idx, countReward(Q), delta))
        print("debug reward", iter_idx, countReward(Q), delta)
    return Q,path,eReward


def q_learning(env, alpha=0.1, gamma=1.0, epsilon=0.1, episodes=10000):
    Q = defaultdict(float)

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # 根据策略生成轨迹的下一个状态
            action = env.epsilon_greedy(Q, state, epsilon)
            #生成轨迹使用的策略是e-greedy 记为策略a
            next_state, reward, done = env.step(state, action)

            # Q-learning更新公式
            #目标策略b
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            #off-policy：根据max q找到下一个action a，并根据a计算reward，a的产生和当前策略无关
            td_target = reward + gamma * max_next_q
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] += alpha * td_error

            state = next_state

    return Q


def print_q_table(env, Q):
    print("Q-table:")
    for i in range(env.rows):
        for j in range(env.cols):
            state = (i, j)
            if state == env.end:
                print(f"State {state} (Terminal)")
                continue
            q_values = [Q[(state, a)] for a in env.actions]
            arrows = [f"{env.action_names[a]}:{q:.2f}" for a, q in zip(env.actions, q_values)]
            print(f"State {state}: {', '.join(arrows)}")
        print("+" + "-" * 50 + "+")


def compare_results(env, sarsa_q, qlearn_q):
    print("\nSARSA策略：")
    print_q_table(env, sarsa_q)

    print("\nQ-learning策略：")
    print_q_table(env, qlearn_q)


if __name__ == "__main__":
    env = GridWorldV4(rows=5,cols=5,terminal_states={(3, 2)}, gamma=0.9)

    # 训练参数
    params = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "episodes": 5000,
        "max_steps":100
    }

    # 训练算法
    sarsa_q, path, eReward = sarsa(env, **params)
    sarsa_p = env.get_policy(sarsa_q)
    #print("debug2", sarsa_p)
    for s in sarsa_p:
        print(s,sarsa_p[s])
    #绘制所有轨迹
    #print(path)
    #plot_epi_num(path, (env.rows,env.cols))
    #绘制error
    plot_delta_curve([item[2] for item in eReward])
    # 绘制reward
    #plot_delta_curve([item[1] for item in eReward])
    env.render_policy(sarsa_p)
    #qlearn_q = q_learning(env, **params)

    # 结果对比
    #compare_results(env, sarsa_q, qlearn_q)