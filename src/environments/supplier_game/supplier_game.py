import time
from typing import Callable

import numpy as np
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)


class SupplierGameState:
    def __init__(self, working_machines, steps_since_last_maintenance, steps_since_last_repair):
        assert len(steps_since_last_maintenance) == len(working_machines) == len(steps_since_last_repair)
        self.working_machines = working_machines
        self.steps_since_last_maintenance = steps_since_last_maintenance
        self.steps_since_last_repair = steps_since_last_repair

    def to_vector(self):
        return self.working_machines + self.steps_since_last_maintenance + self.steps_since_last_repair

    def copy(self):
        return SupplierGameState(
            self.working_machines.copy(),
            self.steps_since_last_maintenance.copy(),
            self.steps_since_last_repair.copy()
        )


class SupplierGame:
    def __init__(
            self,
            screen_width: int,
            screen_height: int,
            num_machines: int,
            maintenance_duration: int,
            repair_duration: int,
            defect_prob_fun: Callable[[int], float],
            state: SupplierGameState
    ):
        self.num_machines = num_machines
        self.state = state
        self.prev_state = state
        self.maintenance_duration = maintenance_duration
        self.repair_duration = repair_duration
        self.defect_prob_fun = defect_prob_fun
        self.screen_width = screen_width
        self.screen_height = screen_height

        actions = [[(i, "no action"), (i, "maintain"), (i, "repair")] for i in range(num_machines)]
        self.actions = [a for a_s in actions for a in a_s]

        # pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Supplier Game')
        self.font = pygame.font.SysFont(None, 48)

    def act(self, action_index: int) -> tuple[float, bool]:
        self.state.steps_since_last_maintenance = [x + 1 for x in self.state.steps_since_last_maintenance]
        self.state.steps_since_last_repair = [x + 1 for x in self.state.steps_since_last_repair]

        action = self.actions[action_index][1]
        machine_index = self.actions[action_index][0]
        if action == "maintain" and self.state.working_machines[machine_index] == 1:
            self.state.working_machines[machine_index] = 0
            self.state.steps_since_last_maintenance[machine_index] = 0
        elif action == "repair" and self.state.working_machines[machine_index] == -1:
            self.state.steps_since_last_repair[machine_index] = 0

        for i, machine in enumerate(self.state.working_machines):
            if machine == 1:
                defect_prob = self.defect_prob_fun(
                    max(self.state.steps_since_last_maintenance[i] - self.maintenance_duration, 0)
                )
                self.state.working_machines[i] = 2 * (np.random.random() > defect_prob) - 1
            elif machine == 0:
                if self.state.steps_since_last_maintenance[i] - self.maintenance_duration == 0:
                    self.state.working_machines[i] = 1
                    self.state.steps_since_last_maintenance[i] = 0
            else:
                if self.state.steps_since_last_repair[i] - self.repair_duration == 0:
                    self.state.working_machines[i] = 1
                    self.state.steps_since_last_repair[i] = 0

        num_working = sum([1 for m in self.state.working_machines if m == 1])
        num_broken = sum([1 for m in self.state.working_machines if m == -1])
        num_maintaining = sum([1 for m in self.state.working_machines if m == 0])

        reward = 0
        reward += num_working * 5  # Reward working machines
        reward -= num_broken * 100  # Heavily penalize broken machines
        reward -= num_maintaining * 1  # Small cost for maintenance

        # Bonus for good states
        if num_broken == 0:
            reward += 20  # Bonus for no broken machines

        return reward, False

    def render(self, iteration: int) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(WHITE)

        num_cols = int(np.floor(np.sqrt(self.num_machines)))
        num_rows = self.num_machines // num_cols
        num_machines_last_row = self.num_machines % num_cols
        machine_width = self.screen_width / num_cols
        machine_height = self.screen_height / (num_rows + min(1, num_machines_last_row))

        for col in range(num_cols):
            for row in range(num_rows):
                machine = self.state.working_machines[row * num_cols + col]
                pygame.draw.rect(
                    self.screen,
                    color=GREEN if machine == 1 else (RED if machine == -1 else ORANGE),
                    rect=(col * machine_width, row * machine_height, machine_width - 2, machine_height - 2))
        for col in range(num_machines_last_row):
            machine = self.state.working_machines[num_rows * num_cols + col]
            pygame.draw.rect(
                self.screen,
                color=GREEN if machine == 1 else (RED if machine == -1 else ORANGE),
                rect=(col * machine_width, (num_rows + 1) * machine_height, machine_width - 2, machine_height - 2))

        # Draw iteration counter
        label_cnt = self.font.render(str(iteration), True, BLACK)
        self.screen.blit(label_cnt, (20, 20))

        pygame.display.flip()
        return True

    def get_actions(self):
        return self.actions

    def with_state(self, state):
        self.state = state
        return self

    def step(self, action_index: int, iteration: int):
        reward, done = self.act(action_index)

        self.render(iteration)
        #time.sleep(.025)

        return self.state.to_vector(), reward, done
