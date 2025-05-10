import numpy as np
import pygame

from src.distances import manhattan_distance

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class GridGameState:
    def __init__(self, agent_pos, diamond_pos, enemies_pos, obstacles_pos):
        self.agent_pos = agent_pos
        self.diamond_pos = diamond_pos
        self.enemies_pos = enemies_pos
        self.obstacles_pos = obstacles_pos

    def to_vector(self):
        return (list(self.agent_pos) +
                list(self.diamond_pos) +
                [item for tup in self.enemies_pos for item in tup] +
                [item for tup in self.obstacles_pos for item in tup])

    def copy(self):
        return GridGameState(self.agent_pos, self.diamond_pos, self.enemies_pos, self.obstacles_pos)


class GridGame:
    def __init__(
            self,
            screen_width: int,
            screen_height: int,
            num_rows: int,
            num_cols: int,
            state: GridGameState
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.width = screen_width / self.num_cols
        self.height = screen_height / self.num_rows
        self.seen = np.zeros((num_cols, num_rows))
        self.state = state
        self.prev_state = state

        # pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Grid Game')
        self.font = pygame.font.SysFont(None, 48)

    def move_agent(self, action_index: int) -> tuple[float, bool]:
        action = ACTIONS[action_index]
        prev_pos = self.prev_state.agent_pos
        current_pos = self.state.agent_pos
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        diamond_pos = self.state.diamond_pos

        self.state.prev_state = self.state.copy()

        if not self.is_pos_valid(next_pos):
            self.seen[current_pos[0], current_pos[1]] += 1
            return -10, False
        else:
            self.state.agent_pos = next_pos
            self.seen[next_pos[0], next_pos[1]] += 1
            if next_pos == prev_pos:
                return -10, False
            if next_pos == self.state.diamond_pos:
                return 100, True
            elif next_pos in self.state.enemies_pos:
                return -100, False
            else:
                distance = manhattan_distance(current_pos, diamond_pos)
                next_distance = manhattan_distance(next_pos, diamond_pos)
                return (distance - next_distance) - 1, False

    def is_pos_valid(self, pos) -> bool:
        return self.num_cols > pos[0] >= 0 and self.num_rows > pos[1] >= 0 and pos not in self.state.obstacles_pos

    def move_enemies_randomly(self):
        enemies_pos = []
        for enemy in self.state.enemies_pos:
            valid_actions = [action for action in ACTIONS if
                             self.is_pos_valid((enemy[0] + action[0], enemy[1] + action[1]))]
            num_actions = len(valid_actions)
            rand_index = np.random.randint(num_actions)
            new_enemy = (enemy[0] + valid_actions[rand_index][0], enemy[1] + valid_actions[rand_index][1])
            enemies_pos.append(new_enemy)
        self.state.enemies_pos = enemies_pos

    def draw_object(self, object_pos, color):
        start_coord = (self.height * object_pos[0], self.width * object_pos[1])
        pygame.draw.rect(self.screen, color,
                         (start_coord[0] + self.height / 4, start_coord[1] + self.width / 4, self.height / 2,
                          self.width / 2))

    def render(self, iteration: int) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(WHITE)

        # Draw board
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                start_coord = (self.width * column + 1, self.height * row + 1)
                pygame.draw.rect(
                    self.screen,
                    (min(255, max(0, self.seen[column][row] - 255)),
                     min(250, max(0, self.seen[column][row] - 510)),
                     min(self.seen[column][row], 255)),
                    (start_coord[0], start_coord[1], self.width - 2, self.height - 2))

        # Draw objects
        self.draw_object(self.state.diamond_pos, GREEN)
        self.draw_object(self.state.agent_pos, RED)
        for obstacle in self.state.obstacles_pos:
            self.draw_object(obstacle, BLUE)
        for enemy in self.state.enemies_pos:
            self.draw_object(enemy, PURPLE)

        # Draw iteration counter
        label_cnt = self.font.render(str(iteration), True, BLACK)
        self.screen.blit(label_cnt, (20, 20))

        pygame.display.flip()
        return True

    def get_actions(self):
        return ACTIONS

    def with_state(self, state):
        self.state = state
        return self

    def step(self, action_index: int, iteration: int):
        reward, done = self.move_agent(action_index)
        self.move_enemies_randomly()

        self.render(iteration)
        # time.sleep(.025)

        return self.state.to_vector(), reward, done
