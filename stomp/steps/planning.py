import random
from typing import List

from tqdm import tqdm

from stomp.foundation import Foundation


class Planning:
    def __init__(
        self,
        foundation: Foundation,
        alpha_step_size: float = 1.0,
    ):
        self.foundation = foundation
        self.alpha_step_size = alpha_step_size

    def plan_with_options(self, num_lookahead_operations: int = 6_000) -> List[float]:
        # We need to iterate over all option (primitive actions included)
        available_options = list(range(self.foundation.num_options))

        # List to save the initial state estimative after planning
        initial_state_planning_estimative = []

        for operation in tqdm(range(num_lookahead_operations)):
            # Get random state
            state = random.choice(
                list(self.foundation.env.state_coordinates_to_idx.keys())
            )
            state_features = self.foundation.env.get_one_hot_state(state)
            max_backup_value = float("-inf")

            # Save initial state estimative
            initial_state_features = self.foundation.env.get_one_hot_state(
                self.foundation.env.initial_state
            )
            initial_state_planning_estimative.append(
                float(self.foundation.w @ initial_state_features)
            )

            # Evaluate each option
            for option in available_options:
                reward = float(self.foundation.w_rewards[option] @ state_features)
                next_state_features = (
                    self.foundation.W_transitions[option] @ state_features
                )
                next_state_value = float(self.foundation.w @ next_state_features)
                backup_value = reward + next_state_value
                max_backup_value = max(max_backup_value, backup_value)

            # With the best option chosen, we now can update the environment weights
            delta = max_backup_value - float(self.foundation.w @ state_features)
            self.foundation.w += self.alpha_step_size * delta * state_features

            # wandb integration
            if self.foundation.wandb_run is not None:
                self.foundation.wandb_run.log(
                    {
                        "planning_initial_state_estimative": initial_state_planning_estimative[-1],
                    }
                )

        return initial_state_planning_estimative
