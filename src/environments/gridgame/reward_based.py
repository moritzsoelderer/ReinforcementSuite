from matplotlib import pyplot as plt
from src.optimization.reward_based_rl import reinforce, perform_episode
from src.environments.gridgame.grid_game import GridGame, GridGameState

AGENT = (1, 1)
DIAMOND = (10, 7)
OBSTACLES = [(3, 0), (3, 1), (3, 2), (3, 3), (2, 4), (12, 7), (12, 6), (12, 5)]
ENEMIES = [(6, 1), (1, 8), (6, 6)]

if __name__ == "__main__":
    init_state = GridGameState(AGENT, DIAMOND, ENEMIES, OBSTACLES)
    game = GridGame(
        600, 500, 10, 12,
        init_state
    )

    num_episodes = 1000
    policy, _, total_rewards = reinforce(game, init_state.copy(), num_episodes=num_episodes, max_iterations=300,
                                         learning_rate=0.0001, layers=[64, 64, 64])

    # Replay episode with trained policy
    perform_episode(game.with_state(init_state), policy, max_iterations=30, sleep=.05)

    plt.plot(range(num_episodes), total_rewards)
    plt.title("Episode Returns")
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Discounted Rewards")
    plt.show()