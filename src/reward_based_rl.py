import time

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Categorical

from RLNet import RLNet
from src.grid_game import GridGame, GridGameState

AGENT = (1, 1)
DIAMOND = (10, 7)
OBSTACLES = [(3, 0), (3, 1), (3, 2), (3, 3), (2, 4), (12, 7), (12, 6), (12, 5)]
ENEMIES = [(6, 1), (1, 8), (6, 6)]


def normalized_discounted_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    return (returns - returns.mean()) / (returns.std() + 1e-8)


def perform_episode(game, policy: RLNet, max_iterations: int = 1000, sleep=None):
    log_probs = []
    entropies = []
    rewards = []

    done = False
    iteration = 0
    while not done:
        if iteration == max_iterations:
            break
        iteration += 1
        state_tensor = torch.tensor(game.state.to_vector(), dtype=torch.float32)
        probs = policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()

        log_probs.append(m.log_prob(action))
        entropies.append(m.entropy())
        state, reward, done = game.step(action.item(), iteration)
        if sleep is not None:
            time.sleep(sleep)
        rewards.append(reward)
   
    return rewards, log_probs, entropies


def reinforce(game, init_state, num_episodes=1000, max_iterations=1000, learning_rate=0.01, discount_factor=0.95, verbose=1, layers=None, entropy_weight=0.01):
    obs_dim = len(init_state.to_vector())
    n_actions = len(game.get_actions())

    policy = RLNet(obs_dim, hidden_dim=layers, output_dim=n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    policy.set_optimiizer(optimizer)

    total_rewards = []
    for episode in range(num_episodes):
        rewards, log_probs, entropies = perform_episode(
            game.with_state(init_state.copy()), policy, max_iterations=max_iterations)

        # Compute discounted returns
        returns = normalized_discounted_returns(rewards, discount_factor)

        # Compute loss and backpropagate
        loss = torch.stack([-log_prob * G for log_prob, G in zip(log_probs, returns)]).sum()
        loss -= entropy_weight * torch.stack(entropies).sum()
        policy.backpropagate(loss)

        # Logging
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        if verbose > 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return policy, optimizer, total_rewards


if __name__ == "__main__":
    init_state = GridGameState(AGENT, DIAMOND, ENEMIES, OBSTACLES)
    game = GridGame(
        600, 500, 10, 12,
        init_state
    )

    num_episodes = 1000
    policy, _, total_rewards = reinforce(game, init_state.copy(), num_episodes=num_episodes, max_iterations=300, learning_rate=0.001, layers=[64, 64, 64])

    # Replay episode with trained policy
    perform_episode(game.with_state(init_state), policy, max_iterations=30, sleep=.05)

    plt.plot(range(num_episodes), total_rewards)
    plt.title("Episode Returns")
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Discounted Rewards")
    plt.show()
