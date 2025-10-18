import random

import torch
from torch import optim, nn

from src.optimization.RLNet import RLNet


def q_learning(game, init_state, num_episodes=1000, max_iterations=1000, learning_rate=0.01, discount_factor=0.95, verbose=1,
               layers=None):
    obs_dim = len(init_state.state.to_vector())
    n_actions = len(game.get_actions())

    q_net = RLNet(obs_dim, hidden_dim=layers, output_dim=n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    q_net.set_optimiizer(optimizer)

    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        game = game.with_state(init_state.copy())
        epsilon = max(0.01, 0.1 * (0.995 ** episode))
        state = torch.tensor(game.state.to_vector(), dtype=torch.float32)

        done = False
        iteration = 1
        rewards = []
        while not done:
            if iteration >= max_iterations:
                break
            iteration += 1
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                q_values = q_net(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done = game.step(action, iteration)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            rewards.append(reward)

            q_next = q_net(next_state).max().detach()
            target = reward + discount_factor * q_next

            # Compute prediction and loss
            pred = q_net(state)[action]
            loss = criterion(pred, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if verbose > 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")
