import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from src.environments.supplier_game.supplier_game import SupplierGameState, SupplierGame
from src.optimization.random_search import random_search, perform_episode

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
    total_rewards = random_search(suppliergame, init_state.copy(), num_episodes=num_episodes,
                                                 max_iterations=max_iterations)

    # Replay episode with trained policy
    perform_episode(suppliergame.with_state(init_state), max_iterations=max_iterations, sleep=.05)

    pd.Series(total_rewards).plot()
    plt.show()
    print("Max Total Rewards:", max(total_rewards))