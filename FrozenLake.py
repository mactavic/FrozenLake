import gym
import time
import numpy as np
import pygame
import sys
from collections import deque


pygame.init()

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 定义网格大小和窗口尺寸
GRID_SIZE = 6
CELL_SIZE = 100
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 50)  # 增加50像素的高度用于显示得分

# 初始化 Pygame 窗口
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("FrozenLake Q-Learning")


path = []  # 用于记录智能体的移动路径



def render_environment(env, agent_pos, score, steps):
    screen.fill(WHITE)


    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, BLACK)
    steps_text = font.render(f"Steps: {steps}", True, BLACK)
    screen.blit(score_text, (10, GRID_SIZE * CELL_SIZE + 10))
    screen.blit(steps_text, (200, GRID_SIZE * CELL_SIZE + 10))


    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if env.desc[i, j] == b'H':  # 渲染陷阱
                pygame.draw.rect(screen, RED, rect)
            elif env.desc[i, j] == b'G':  # 渲染目标
                pygame.draw.rect(screen, GREEN, rect)


    if len(path) >= 2:
        for i in range(1, len(path)):
            start_pos = (path[i - 1][1] * CELL_SIZE + CELL_SIZE // 2, path[i - 1][0] * CELL_SIZE + CELL_SIZE // 2)
            end_pos = (path[i][1] * CELL_SIZE + CELL_SIZE // 2, path[i][0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.line(screen, BLUE, start_pos, end_pos, 3)


    pygame.draw.circle(screen, BLUE, (agent_pos[1] * CELL_SIZE + CELL_SIZE // 2, agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                       CELL_SIZE // 4)

    pygame.display.flip()



def update_path(agent_pos):
    path.append(agent_pos)  # 记录当前位置
    if len(path) > 100:  # 限制路径长度，避免内存占用过多
        path.pop(0)


# 检查地图是否有有效路径
def has_valid_path(map_layout):
    rows, cols = len(map_layout), len(map_layout[0])
    start = (0, 0)  # 起点 S
    goal = (rows - 1, cols - 1)  # 目标点 G

    # 使用 BFS 检查路径
    queue = deque([start])
    visited = set()
    visited.add(start)

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True  # 找到目标点

        # 遍历四个方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and map_layout[nx][ny] != 'H':
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False  # 未找到目标点



def generate_random_map():
    while True:
        # 初始化地图
        map_layout = [['F' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        map_layout[0][0] = 'S'  # 起点
        map_layout[GRID_SIZE - 1][GRID_SIZE - 1] = 'G'  # 目标点

        # 随机放置陷阱
        trap_count = 15  # 改为15个陷阱
        for _ in range(trap_count):
            while True:
                x, y = np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)
                if map_layout[x][y] == 'F':  # 确保不覆盖起点和目标点
                    map_layout[x][y] = 'H'
                    break

        # 检查是否有有效路径
        if has_valid_path(map_layout):
            return [''.join(row) for row in map_layout]  # 转换为字符串列表


class QLearning:
    def __init__(self, n_states, n_actions, epsilon, gamma, lr, epsilon_decay, min_epsilon):
        self.n_states = n_states  # 状态数
        self.n_actions = n_actions  # 动作数
        self.epsilon = epsilon  # 贪婪策略中的\epsilon
        self.gamma = gamma  # 衰减因子
        self.lr = lr  # 学习率
        self.epsilon_decay = epsilon_decay  # \epsilon的衰减程度
        self.min_epsilon = min_epsilon  # \epsilon的最小值
        self.qtable = self.build_q_table()  # Q-table

    def build_q_table(self):  # 初始化一个全0的Q-table
        return np.zeros((self.n_states, self.n_actions))

    def choose_action(self, obs):  # 根据\epsilon - greedy策略选择动作
        if np.random.uniform() >= self.epsilon:
            Q_max = np.max(self.qtable[obs])
            action_list = np.where(self.qtable[obs] == Q_max)[0]  # 最大值可能不止一个
            action = np.random.choice(action_list)
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn_q_table(self, s, a, r, s_, done):  # 更新Q-table
        q_predict = self.qtable[s][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.qtable[s_])
        self.qtable[s][a] += self.lr * (q_target - q_predict)

    def update_epsilon(self):  # 随着次数增加，探索的概率应不断降低
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':

    custom_map = generate_random_map()
    print("Generated Map:")
    for row in custom_map:
        print(row)


    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False)  # 使用自定义地图
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=0.95,
        gamma=0.9,
        lr=0.1,
        epsilon_decay=0.95,
        min_epsilon=0.01
    )

    episodes = 50000
    render = False  # 训练时不渲染


    for episode in range(episodes):
        total_reward, total_steps = 0, 0
        s, _ = env.reset()

        while True:
            a = agent.choose_action(s)
            s_, r, done, truncated, _ = env.step(a)


            if done:
                if env.desc[s_ // GRID_SIZE, s_ % GRID_SIZE] == b'G':  # 到达目标
                    total_reward += 10
                elif env.desc[s_ // GRID_SIZE, s_ % GRID_SIZE] == b'H':  # 掉入陷阱
                    total_reward -= 10
            total_reward -= 1  # 每移动一步扣1分

            agent.learn_q_table(s, a, r, s_, done)
            s = s_
            total_steps += 1

            if done or truncated:
                break

        if episode % 5 == 0:
            agent.update_epsilon()

        print('Episode {:03d} | Step:{:03d} | Reward:{:.03f}'.format(episode, total_steps, total_reward))


    render = True  # 开启渲染
    s, _ = env.reset()
    agent_pos = (s // GRID_SIZE, s % GRID_SIZE)  # 将状态转换为网格坐标
    total_reward = 0  # 初始化总得分
    total_steps = 0  # 初始化总步数

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if render:
            render_environment(env, agent_pos, total_reward, total_steps)
            time.sleep(0.5)  # 控制渲染速度

        a = agent.choose_action(s)
        s_, r, done, truncated, _ = env.step(a)


        if done:
            if env.desc[s_ // GRID_SIZE, s_ % GRID_SIZE] == b'G':  # 到达目标
                total_reward += 10
            elif env.desc[s_ // GRID_SIZE, s_ % GRID_SIZE] == b'H':  # 掉入陷阱
                total_reward -= 10
        total_reward -= 1  # 每移动一步扣1分

        s = s_
        agent_pos = (s // GRID_SIZE, s % GRID_SIZE)  # 更新智能体位置
        update_path(agent_pos)  # 更新路径
        total_steps += 1  # 更新步数

        if done or truncated:
            print("Test Episode | Reward: {:.03f} | Steps: {}".format(total_reward, total_steps))
            # 保持窗口显示，直到用户关闭
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                render_environment(env, agent_pos, total_reward, total_steps)
                time.sleep(0.1)