import random
import time
from collections import Counter
from environment import TwoSnakeEnv  # 确保和environment.py在同一目录


def test_random_play(num_episodes: int = 100000, render_first_n: int = 3):
    env = TwoSnakeEnv()

    wins = Counter({'A': 0, 'B': 0, 'draw': 0})
    total_steps = 0
    total_length_winner = 0
    max_steps_seen = 0

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state_a, state_b = env.reset()
        done = False
        steps = 0

        # 前几局可视化（方便肉眼检查逻辑是否正确）
        if episode <= render_first_n:
            print(f"\n=== Episode {episode} (可视化) ===")
            env.render()  # 打印初始状态

        while not done:
            # 纯随机相对动作（0=left, 1=straight, 2=right）
            action_a = random.randint(0, 2)
            action_b = random.randint(0, 2)

            ns_a, ns_b, r_a, r_b, done, info = env.step(action_a, action_b)
            steps += 1

            if episode <= render_first_n:
                env.render()
                print(f"Actions: A={['left', 'straight', 'right'][action_a]} "
                      f"B={['left', 'straight', 'right'][action_b]} | "
                      f"Reward A={r_a:.1f} B={r_b:.1f}")
                print("-" * 50)

        winner = info['winner']
        wins[winner] += 1
        total_steps += steps
        max_steps_seen = max(max_steps_seen, steps)

        if winner != 'draw':
            winner_len = info['length_a'] if winner == 'A' else info['length_b']
            total_length_winner += winner_len

        if episode % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"已完成 {episode}/{num_episodes} 局 | "
                  f"用时 {elapsed:.1f}s | "
                  f"当前胜率 A:{wins['A'] / episode * 100:5.2f}% "
                  f"B:{wins['B'] / episode * 100:5.2f}% "
                  f"平局:{wins['draw'] / episode * 100:5.2f}%")

    # 最终统计
    total_episodes = num_episodes
    win_rate_a = wins['A'] / total_episodes * 100
    win_rate_b = wins['B'] / total_episodes * 100
    draw_rate = wins['draw'] / total_episodes * 100
    avg_steps = total_steps / total_episodes
    avg_winner_length = total_length_winner / (wins['A'] + wins['B']) if (wins['A'] + wins['B']) > 0 else 0

    print("\n" + "=" * 60)
    print("随机策略 10000局测试完成！")
    print("=" * 60)
    print(f"A 胜率     : {win_rate_a:.2f}%")
    print(f"B 胜率     : {win_rate_b:.2f}%")
    print(f"平局率     : {draw_rate:.2f}%")
    print(f"平均回合步数 : {avg_steps:.1f}")
    print(f"胜者平均长度 : {avg_winner_length:.1f}")
    print(f"最长回合    : {max_steps_seen} 步")
    print(f"总用时      : {time.time() - start_time:.1f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    # 跑1万局大概只需要15-25秒（极快）
    test_random_play(num_episodes=100000, render_first_n=2)  # 前2局会打印地图