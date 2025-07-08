
import os
import pickle
from os.path import join
from typing import Dict

from stomp.steps.planning import TemporaryPlanning as Planning
from gridworld.gridworld import TemporaryGridWorld as GridWorld, State
from stomp.foundation import TemporaryFoundation as Foundation
from stomp.steps.model_learning import TemporaryModelLearning as ModelLearning
from stomp.steps.option_learning import TemporaryOptionLearning as OptionLearning


class TemporarySTOMP:
    def __init__(
        self,
        env: GridWorld,
        subgoal_states_info: Dict[int, State],
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_prime: float = 0.1,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        alpha_step_size: float = 1.0,
        lambda_: float = 0,
        lambda_prime: float = 0,
        experiment_results_path: str | None = None,
    ):
        self.experiment_results_path = experiment_results_path
        self.subagoals_state_idx = list(subgoal_states_info.keys())
        self.subgoals_state = list(subgoal_states_info.values())

        self.stomp_foundation = Foundation(
            env=env,
            subgoals_state=self.subgoals_state,
            subgoals_state_idx=self.subagoals_state_idx,
            gamma=gamma,
        )

        self.option_learning = OptionLearning(
            foundation=self.stomp_foundation,
            alpha=alpha,
            alpha_prime=alpha_prime,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        self.model_learning = ModelLearning(
            foundation=self.stomp_foundation,
            alpha_r=alpha_r,
            alpha_p=alpha_p,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        self.planning = Planning(
            foundation=self.stomp_foundation,
            alpha_step_size=alpha_step_size,
        )

    def execute(
        self,
        num_lookahead_operations: int = 6_000,
        off_policy_steps: int = 50_000,
        experiment_folder_prefix: str = "stomp",
    ):
        print("[INFO] Starting STOMP execution...\n")

        option_learning_logs = []
        model_learning_logs = []

        for subgoal_idx in range(len(self.subagoals_state_idx)):
            print(
                f"\n[INFO] Learning options for subgoal {subgoal_idx + 1}/{self.stomp_foundation.num_subgoals}"
            )
            initial_state_estimative, rmse_of_state_values = self.option_learning.learn_options(
                subgoal_idx, off_policy_steps, return_rmse=True
            )
            option_learning_logs.append((initial_state_estimative, rmse_of_state_values))

        for option_idx in range(self.stomp_foundation.num_options):
            print(
                f"\n[INFO] Learning model for option {option_idx + 1}/{self.stomp_foundation.num_options}, {'a Primitive Action' if option_idx < self.stomp_foundation.env.num_actions else 'a Full Option'}"
            )
            reward_model_rmses, transition_model_errors = (
                self.model_learning.learn_model(option_idx, off_policy_steps)
            )
            model_learning_logs.append((reward_model_rmses, transition_model_errors))

        print("\n[INFO] Planning with learned options and models...")
        planning_logs = self.planning.plan_with_options(num_lookahead_operations)

        if self.experiment_results_path is not None:
            save_files_path = join(
                self.experiment_results_path, experiment_folder_prefix
            )
            print("\n[INFO] Saving Files...")
            os.makedirs(save_files_path, exist_ok=True)
            self.stomp_foundation.env.save_room(save_files_path)
            self.stomp_foundation.save_vectors(save_files_path)
            with open(
                join(save_files_path, "option_learning_logs.pkl"),
                "wb",
            ) as f:
                pickle.dump(option_learning_logs, f)

            with open(
                join(save_files_path, "model_learning_logs.pkl"),
                "wb",
            ) as f:
                pickle.dump(model_learning_logs, f)

            with open(join(save_files_path, "planning_logs.pkl"), "wb") as f:
                pickle.dump(planning_logs, f)
            print(f"\n[INFO] Files saved on {save_files_path}")

        return option_learning_logs, model_learning_logs, planning_logs
