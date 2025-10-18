from src.environments.gridgame.grid_game import GridGame, GridGameState
from src.optimization.state_based_rl import q_learning

AGENT = (1, 1)
DIAMOND = (10, 7)
OBSTACLES = [(3, 0), (3, 1), (3, 2), (3, 3), (2, 4), (12, 7), (12, 6), (12, 5)]
ENEMIES = [(6, 1), (1, 8), (6, 6)]

if __name__ == '__main__':
    init_state = GridGameState(AGENT, DIAMOND, ENEMIES, OBSTACLES)

    game = GridGame(
        600, 500, 10, 12,
        init_state
    )
    q_learning(game.with_state(init_state), init_state)
    