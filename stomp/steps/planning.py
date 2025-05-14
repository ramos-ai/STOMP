import numpy as np
from tqdm import tqdm

from common.utils import get_progress_bar
from gridworld.gridworld import GridWorld
from stomp.foundation import STOMPFoundation


class Planning(STOMPFoundation):
    def __init__(
        self,
        env: GridWorld,
        gamma: np.float64 = 0.99,
        alpha_step_size: np.float64 = 1.0,
    ):
        super().__init__(env, gamma)
        self.alpha_step_size = alpha_step_size

    def plan_with_options(
        self, num_lookahead_operations: int = 6_000, log_freq: int = 100
    ):
        # We need to iterate over all option (primitive actions included)
        available_options = list(range(self.num_options))

        # List to save the initial state estimative after planning
        initial_state_planning_estimative = []

        progress_bar = (
            range(num_lookahead_operations)
            if log_freq is None
            else get_progress_bar(num_lookahead_operations)
        )

        for operation in progress_bar:
            # Get a random state
            state = self.env.get_random_state()

            # To reproduce Fig 1 we need to plan until reach the hallway,
            # therefore, if the random state chosen is the hallway state,
            # we try to get a different state.
            # hallway_state = np.where(self.env.room_array == 8)
            # while state == hallway_state:
            #     state = self.env.get_random_state()

            state_features = self.env.state_to_features(state)
            max_backup_value = float("-inf")

            v = self.compute_gvf()
            v_0 = (self.env.initial_state[0][0], self.env.initial_state[1][0])
            initial_state_planning_estimative.append(v[v_0])

            # Evaluating all options
            for option in available_options:
                reward = self.w_rewards[option] @ state_features
                next_state_features = self.W_transitions[option] @ state_features
                next_state_value = self.w @ next_state_features
                backup_value = reward + next_state_value
                max_backup_value = max(max_backup_value, backup_value)

            # With the best option chosen, we now can update the environment weights
            delta = max_backup_value - self.w @ state_features
            self.w += self.alpha_step_size * delta * state_features

            if log_freq is not None and operation % log_freq == 0:
                tqdm.write(
                    f"Operation {operation}: Max backup value: {max_backup_value:.4f}, Delta: {delta:.4f}, Initial State Estimative: {initial_state_planning_estimative[-1]:.4f}, Updated weights sum: {self.w.sum():.4f}."
                )

        return initial_state_planning_estimative
