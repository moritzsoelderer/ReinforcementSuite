import random

import torch
from torch import optim, nn

from src.RLNet import RLNet
from src.grid_game import GridGame, GridGameState

AGENT = (1, 1)
DIAMOND = (10, 7)
OBSTACLES = [(3, 0), (3, 1), (3, 2), (3, 3), (2, 4), (12, 7), (12, 6), (12, 5)]
ENEMIES = [(6, 1), (1, 8), (6, 6)]

def q_learning():
    init_state = GridGameState(AGENT, DIAMOND, ENEMIES, OBSTACLES)

    game = GridGame(
        600, 500, 10, 12,
        init_state
    )

    learning_rate = 0.005
    obs_dim = len(init_state.to_vector())
    n_actions = len(game.get_actions())

    q_net = RLNet(obs_dim, hidden_dim=64, output_dim=n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    q_net.set_optimiizer(optimizer)

    num_episodes = 1000
    gamma = 0.95
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        game = game.with_state(init_state.copy())
        epsilon = max(0.01, 0.1 * (0.995 ** episode))
        state = torch.tensor(game.state.to_vector(), dtype=torch.float32)

        done = False
        iteration = 1
        rewards = []
        while not done:
            if iteration >= 1000:
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
            target = reward + gamma * q_next

            # Compute prediction and loss
            pred = q_net(state)[action]
            loss = criterion(pred, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        print(f"Episode {episode}, Total Reward: {sum(rewards)}")



if __name__ == '__main__':
    q_learning()
