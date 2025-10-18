from datetime import datetime

from sympy.printing.pytorch import torch

from src.optimization.reward_based_rl import reinforce, perform_episode
from src.environments.singleplayerpong.single_player_pong import SinglePlayerPong, SinglePlayerPongState

if __name__ == "__main__":
    init_state = SinglePlayerPongState(agent_pos=(60, 400), agent_height=100,
                                       ball_pos=(580, 400), ball_vel=(-10, -10), points=(0, 0))

    pong = SinglePlayerPong(
        screen_width=800, screen_height=600, init_state=init_state
    )

    num_episodes = 10000
    max_iterations = 1000000
    policy, optimizer, total_rewards = reinforce(pong, init_state.copy(), num_episodes=num_episodes,
                                                 max_iterations=max_iterations,
                                                 learning_rate=0.00001, discount_factor=0.96, layers=[256, 128, 64],
                                                 entropy_weight=0.1)

    # Replay episode with trained policy
    perform_episode(pong.with_state(init_state), policy, max_iterations=max_iterations, sleep=.05)

    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'SinglePlayerPong_' + datetime.now().strftime('%m%d%Y_%H%M%S') + '.pth')
