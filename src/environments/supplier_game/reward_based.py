from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
from sympy.printing.pytorch import torch
import pandas as pd

from src.optimization.reward_based_rl import reinforce, perform_episode
from src.environments.supplier_game.supplier_game import SupplierGameState, SupplierGame

if __name__ == "__main__":
    matplotlib.use('TkAgg')
    init_state = SupplierGameState(
        working_machines=[1, 1],
        steps_since_last_maintenance=[0, 0],
        steps_since_last_repair=[0, 0]
    )

    suppliergame = SupplierGame(
        screen_width=800, screen_height=600, state=init_state,
        num_machines=2, defect_prob_fun=lambda n: n*0.25,
        maintenance_duration=0, repair_duration=5
    )

    num_episodes = 3000
    max_iterations = 50
    policy, optimizer, total_rewards = reinforce(suppliergame, init_state.copy(), num_episodes=num_episodes,
                                                 max_iterations=max_iterations,
                                                 learning_rate=0.0001, discount_factor=0.99, layers=[512, 256, 128, 64],
                                                 entropy_weight=0.2)

    # Replay episode with trained policy
    perform_episode(suppliergame.with_state(init_state), policy, max_iterations=max_iterations, sleep=.05)

    pd.Series(total_rewards).plot()
    plt.show()
    print("Max Total Rewards:", max(total_rewards))

    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'SupplierGame_' + datetime.now().strftime('%m%d%Y_%H%M%S') + '.pth')
