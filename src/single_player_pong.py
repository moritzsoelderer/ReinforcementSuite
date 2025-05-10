import time

import numpy as np
import pygame

from src import distances

WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
LIGHT_PINK = (255, 182, 193, 0.5)
RED = (255, 0, 0)
AGENT_HEIGHT_PERC = 0.15

ACTIONS = [(0, 0), (0, 10), (0, -10)]

GOAL_LINE_AGENT = 30


class SinglePlayerPongState:
    def __init__(self, agent_pos, agent_height, ball_pos, ball_vel, points):
        self.agent_pos = agent_pos
        self.agent_height = agent_height
        self.ball_pos = ball_pos
        self.ball_vel = ball_vel
        self.points = points
        
    def to_vector(self):
        return [self.agent_pos[0], self.agent_pos[1], self.agent_height,
                self.ball_pos[0], self.ball_pos[1],
                self.ball_vel[0], self.ball_vel[1], self.points[0], self.points[1]]
    
    def copy(self):
        return SinglePlayerPongState(self.agent_pos, self.agent_height, self.ball_pos, self.ball_vel, self.points)


class SinglePlayerPong:
    def __init__(self, screen_width, screen_height, init_state):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.state = init_state.copy()
        self.init_state = init_state.copy()

        # pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Single Player Pong')
        self.font = pygame.font.SysFont(None, 48)

        np.random.seed(43)

    def move_ball(self):
        self.state.ball_pos = (self.state.ball_pos[0] + self.state.ball_vel[0], self.state.ball_pos[1] + self.state.ball_vel[1])

        paddle_left = self.state.agent_pos[0] - 5
        paddle_right = self.state.agent_pos[0] + 5
        paddle_top = self.state.agent_pos[1] - self.state.agent_height / 2
        paddle_bottom = self.state.agent_pos[1] + self.state.agent_height / 2

        ball_x, ball_y = self.state.ball_pos
        ball_radius = 10

        if paddle_left <= ball_x <= paddle_right and paddle_top <= ball_y <= paddle_bottom:
            self.state.ball_vel = (self.state.ball_vel[0] * -1, self.state.ball_vel[1])
            return 1000

        if ball_y - ball_radius <= 0 or ball_y + ball_radius >= self.screen_height:
            self.state.ball_vel = (self.state.ball_vel[0], self.state.ball_vel[1] * -1)
            return 10

        if ball_x - ball_radius <= GOAL_LINE_AGENT:
            self.state.points = (self.state.points[0], self.state.points[1] + 1)
            self.state.ball_pos = (self.init_state.ball_pos[0], (self.screen_height - 20) * np.random.random() + 10)
            self.state.ball_vel = 10 * (self.init_state.ball_vel[0], 2 * np.random.random() - 1)
            return - 1000

        if ball_x + ball_radius >= self.screen_width - GOAL_LINE_AGENT:
            self.state.points = (self.state.points[0] + 1, self.state.points[1])
            self.state.ball_pos = (self.init_state.ball_pos[0], (self.screen_height - 20) * np.random.random() + 10)
            self.state.ball_vel = 10 * (self.init_state.ball_vel[0], 2 * np.random.random() - 1)
            return + 1000

        return 10

    def move_agent(self, action_index: int):
        next_pos = (self.state.agent_pos[0] + ACTIONS[action_index][0], self.state.agent_pos[1] + ACTIONS[action_index][1])
        if self.state.agent_height * 0.5 <= next_pos[1] <= self.screen_height - self.state.agent_height * 0.5:
            self.state.agent_pos = next_pos

        if self.state.points[0] == 20:
            return 10000, True
        elif self.state.points[1] == 20:
            return -10000, True
        else:
            agent_center_y = next_pos[1]
            ball_y = self.state.ball_pos[1]
            distance_penalty = -np.abs(agent_center_y - ball_y) / (self.screen_height / 2)
            return distance_penalty, False

    def with_state(self, state):
        self.state = state
        return self

    def step(self, action_index: int, iteration: int):
        reward, done = self.move_agent(action_index)
        add_reward = self.move_ball()
        self.render(iteration)

        return self.state.to_vector(), reward + add_reward, done

    def render(self, iteration: int):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(BLACK)

        # Draw Background Texture
        pygame.draw.rect(
            self.screen, RED,
            (0, self.screen_height / 2 - 5, self.screen_width, 10)
        )
        pygame.draw.circle(self.screen, color=RED,
                           center=(self.screen_width / 2, self.screen_height / 2), radius=self.screen_height / 6)
        pygame.draw.circle(self.screen, color=BLACK,
                           center=(self.screen_width / 2, self.screen_height / 2), radius=self.screen_height / 6 - 10)

        # Draw Ball
        pygame.draw.rect(
            self.screen, WHITE,
            (self.state.ball_pos[0] - 10, self.state.ball_pos[1] - 10, 20, 20)
        )

        # Draw Agent
        pygame.draw.rect(
            self.screen, WHITE,
            (self.state.agent_pos[0] - 5, self.state.agent_pos[1] - 0.5 * self.state.agent_height,
             10, self.state.agent_height)
        )

        # Draw goals
        pygame.draw.rect(
            self.screen, LIGHT_PINK,
            (0, 0, GOAL_LINE_AGENT, self.screen_height)
        )
        pygame.draw.rect(
            self.screen, LIGHT_PINK,
            (self.screen_width - GOAL_LINE_AGENT, 0, GOAL_LINE_AGENT, self.screen_height)
        )

        # Draw iteration counter
        label_cnt = self.font.render(str(iteration), True, WHITE)
        self.screen.blit(label_cnt, (self.screen_width * 0.5 - 100, 20))

        # Draw Points
        label_points = self.font.render(str(self.state.points), True, WHITE)
        self.screen.blit(label_points, (self.screen_width * 0.5 + 50, 20))

        pygame.display.flip()
        
    def get_actions(self):
        return ACTIONS
