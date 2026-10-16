# environment.py->我们构建物理世界
import numpy as np
import torch
import random
from collections import deque
from typing import Tuple
from config import GRID_SIZE, WIN_REWARD, LOSS_REWARD, FOOD_REWARD, LENGTH_BONUS, DRAW_REWARD

# 我们规定每一步在网格坐标中的移动
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Only self/opponent channel -> zero-bias parameter-sharing self-play
class Snake:
    def __init__(self, head_pos: Tuple[int, int], direction: int):
        self.direction = direction
        opp_dx, opp_dy = DIRECTIONS[(direction + 2) % 4]  # 头后面2单位为身体(初始化)
        self.body = deque()  # The left side is the head, and the right side is the tail.
        cx, cy = head_pos
        for i in range(3):
            self.body.append((cx, cy))
            cx += opp_dx
            cy += opp_dy

    # 对列第一位是头
    def get_head(self):
        return self.body[0]

    # 通过坐标取得地动后新头的位置
    def get_new_head(self):
        dx, dy = DIRECTIONS[self.direction]
        hx, hy = self.get_head()
        return hx + dx, hy + dy

    # 修改身体坐标以实现移动
    def move(self, new_head, grow: bool):
        self.body.appendleft(new_head)
        if not grow:
            self.body.pop()

    # 对应Action
    # rel_action = 0 → 左转 90°
    # rel_action = 1 → 直行（什么都不做）
    # rel_action = 2 → 右转 90°
    # 如何计算? -> 我们通过 +1/-1 % 4 取得他在DIRECTIONS数组中对应的移动表现
    def turn(self, rel_action: int):
        if rel_action == 0:  # turn left
            self.direction = (self.direction - 1) % 4
        elif rel_action == 2:  # turn right
            self.direction = (self.direction + 1) % 4


class TwoSnakeEnv:
    def __init__(self):
        self.reset()

    # 初始化环境。随机生成两条蛇的位置和方向，确保它们开局不重叠，并随机生成第一个食物。
    def reset(self):
        # while True: 防止“落地成盒”
        while True:
            # 两个头的位置, 安全区域生成 (Safe Spawning), 给蛇留出“展开身体”的空间
            ha = (random.randint(5, GRID_SIZE - 6), random.randint(5, GRID_SIZE - 6))
            hb = (random.randint(5, GRID_SIZE - 6), random.randint(5, GRID_SIZE - 6))
            # 3 个方向随机生成. 方向随机：两条蛇的初始朝向（上/下/左/右）是随机的。
            da = random.randint(0, 3)
            db = random.randint(0, 3)
            self.snake_a = Snake(ha, da)
            self.snake_b = Snake(hb, db)
            # 设置好了之后退出
            if len(set(self.snake_a.body) & set(self.snake_b.body)) == 0:
                break
        # 初始化环境状态
        self.food = self._spawn_food()
        self.steps = 0
        return self._get_observation('A'), self._get_observation('B')

    # 在空位置生成食物
    def _spawn_food(self):
        occupied = set(self.snake_a.body) | set(self.snake_b.body)
        empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) not in occupied]
        return random.choice(empty)

    # Step 函数: 这是一个判决函数
    # refer 文档状态转移函数
    # step 函数就是上帝视角、物理引擎和裁判的结合体。
    # 先转方向 → 计算新头 → 墙优先判死 → 吃食物 → 移动身体 → 生成新食物 → 再检查身体 / 头对头碰撞 → 结算奖励
    def step(self, action_a: int, action_b: int):
        # 单纯步数+1，记录 episode 活了多久
        self.steps += 1

        # 先转方向(左,右,不动)，再计算新头位置
        self.snake_a.turn(action_a)
        self.snake_b.turn(action_b)

        new_head_a = self.snake_a.get_new_head()
        new_head_b = self.snake_b.get_new_head()

        # 墙碰撞优先 : 撞墙 = 直接死亡，对方直接赢
        if not (0 <= new_head_a[0] < GRID_SIZE and 0 <= new_head_a[1] < GRID_SIZE):
            len_a = len(self.snake_a.body)
            len_b = len(self.snake_b.body)
            obs_a = self._get_observation('A')
            obs_b = self._get_observation('B')
            info = {'steps': self.steps, 'length_a': len_a, 'length_b': len_b, 'winner': 'B'}
            return obs_a, obs_b, LOSS_REWARD, WIN_REWARD + LENGTH_BONUS * len_b, True, info

        if not (0 <= new_head_b[0] < GRID_SIZE and 0 <= new_head_b[1] < GRID_SIZE):
            len_a = len(self.snake_a.body)
            len_b = len(self.snake_b.body)
            obs_a = self._get_observation('A')
            obs_b = self._get_observation('B')
            info = {'steps': self.steps, 'length_a': len_a, 'length_b': len_b, 'winner': 'A'}
            return obs_a, obs_b, WIN_REWARD + LENGTH_BONUS * len_a, LOSS_REWARD, True, info

        # 吃食物处理: 同时到达同一食物，随机给一个蛇吃
        eat_a = new_head_a == self.food
        eat_b = new_head_b == self.food
        if eat_a and eat_b:
            eat_a, eat_b = (True, False) if random.random() < 0.5 else (False, True)

        # 执行移动（此时才真正更新身体！）
        self.snake_a.move(new_head_a, eat_a)  # eat_a=True 就不 pop 尾巴 → 长1
        self.snake_b.move(new_head_b, eat_b)

        # 吃到食物就重新生成一个（空位随机）
        if eat_a or eat_b:
            self.food = self._spawn_food()

        # 真正执行碰撞检测
        done, winner = self._check_collisions(new_head_a, new_head_b)

        # 奖励计算
        reward_a = FOOD_REWARD if eat_a else 0.0
        reward_b = FOOD_REWARD if eat_b else 0.0

        len_a = len(self.snake_a.body)
        len_b = len(self.snake_b.body)

        if done:
            length_a = len(self.snake_a.body)
            length_b = len(self.snake_b.body)
            if winner == 'A':
                reward_a += WIN_REWARD + LENGTH_BONUS * length_a
                reward_b += LOSS_REWARD
            elif winner == 'B':
                reward_a += LOSS_REWARD
                reward_b += WIN_REWARD + LENGTH_BONUS * length_b
            else:  # draw
                reward_a += DRAW_REWARD
                reward_b += DRAW_REWARD

        info = {'steps': self.steps, 'length_a': len_a, 'length_b': len_b, 'winner': winner or None}

        return self._get_observation('A'), self._get_observation('B'), reward_a, reward_b, done, info

    # _check_collisions 里严格按顺序：
    # 自撞（包括咬到自己没动的尾巴）
    # 撞对手身体（对手头也算身体）
    # 头对头 → 长度决胜（长吃短，平局双方死）
    def _check_collisions(self, head_a, head_b):
        a_dies = False
        b_dies = False

        # 自撞
        if head_a in list(self.snake_a.body)[1:]:
            a_dies = True
        if head_b in list(self.snake_b.body)[1:]:
            b_dies = True

        # 互撞身体
        if head_a in self.snake_b.body:
            a_dies = True
        if head_b in self.snake_a.body:
            b_dies = True

        # head-to-head（长度决胜）
        if head_a == head_b:
            la = len(self.snake_a.body)  # 注意：此时还没move完，如果吃食物要+1
            lb = len(self.snake_b.body)
            if la > lb:
                b_dies = True
            elif lb > la:
                a_dies = True
            else:
                a_dies = b_dies = True

        if a_dies and b_dies:
            return True, 'draw'
        elif a_dies:
            return True, 'B'
        elif b_dies:
            return True, 'A'
        else:
            return False, None

    # 构建神经网络的输入张量
    # space: 8*30*30
    # 通道定义:
    # Ch 0: 自身身体（不含头） 。
    # Ch 1: 自身头部 。
    # Ch 2: 对手身体（含头） 。
    # Ch 3: 食物 。
    # Ch 4-7: 自身头部方向的 One-hot 编码 。
    def _get_observation(self, perspective: str) -> torch.Tensor:
        # 1. 初始化张量: 创建一个全零的 3D 数组（张量）
        obs = np.zeros((8, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        # 2. 确定视角:
        # 如果请求的是 A 的视角，那么 own 就是蛇 A，opp 就是蛇 B。
        # 如果请求的是 B 的视角，那么 own 就是蛇 B，opp 就是蛇 A。
        own = self.snake_a if perspective == 'A' else self.snake_b
        opp = self.snake_b if perspective == 'A' else self.snake_a
        # 3. 填充第 1 层：自己的身体（不含头）refer: 对应文档: Channel 1: Own body (except head) 。
        for pos in list(own.body)[1:]:
            obs[0, pos[0], pos[1]] = 1.0
        # 4. 填充第 2 层：自己的头部 refer : 对应文档: Channel 2: Own head。
        # 将头和身体分开，是为了让卷积神经网络（CNN）明确知道“头在哪里”，这对于决策至关重要。
        hx, hy = own.get_head()
        obs[1, hx, hy] = 1.0
        # 5. 填充第 3 层：对手的身体（含头）refer: 对应文档: Channel 3: Opponent body (including head) 。
        for pos in opp.body:
            obs[2, pos[0], pos[1]] = 1.0
        # 6. 填充第 4 层：食物位置 refer: 对应文档: Channel 4: Food。
        fx, fy = self.food
        obs[3, fx, fy] = 1.0
        # 7. 填充第 5-8 层：当前朝向 (One-Hot) refer: 文档Channel 3 to 8
        # 含义: 标记“自己”当前的朝向
        # 作用: 告诉神经网络当前蛇头正朝向哪个方向，这对于理解“左转”或“右转”会带来什么后果至关重要（因为动作是相对的）。
        obs[4 + own.direction, hx, hy] = 1.0
        return torch.from_numpy(obs)

    # Terminal以字符的形式绘制当前的游戏画面(给人看)
    def render(self):
        grid = np.full((GRID_SIZE, GRID_SIZE), '.', dtype=object)
        for i, pos in enumerate(self.snake_a.body):
            grid[pos[0], pos[1]] = 'A' if i == 0 else 'a'
        for i, pos in enumerate(self.snake_b.body):
            grid[pos[0], pos[1]] = 'B' if i == 0 else 'b'
        fx, fy = self.food
        grid[fx, fy] = '*'
        print('\n'.join(' '.join(row) for row in grid))
        print(f"Steps: {self.steps}  LenA: {len(self.snake_a.body)}  LenB: {len(self.snake_b.body)}\n")
