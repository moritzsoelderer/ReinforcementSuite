import time

import torch


def perform_episode(env, max_iterations: int = 1000, sleep=None):
    rewards = []
    actions = env.get_actions()
    num_available_actions = len(actions)

    done = False
    iteration = 0
    while not done:
        if iteration == max_iterations:
            break
        iteration += 1
        action = torch.randint(num_available_actions, (1,)).int()
        state, reward, done = env.step(action, iteration)
        if sleep is not None:
            time.sleep(sleep)
        rewards.append(reward)

    return rewards


def random_search(env, init_state, num_episodes=1000, max_iterations=1000, verbose=1):
    total_rewards = []
    for episode in range(num_episodes):
        rewards = perform_episode(
            env.with_state(init_state.copy()), max_iterations=max_iterations)

        total_reward = sum(rewards)
        total_rewards.append(total_reward)

        if verbose > 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return total_rewards
