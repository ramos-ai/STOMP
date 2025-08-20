import os
import pickle
import uuid
from os.path import join
from typing import Dict

import wandb
from gridworld.gridworld import GridWorld, State
from stomp.foundation import Foundation
from stomp.steps.model_learning import ModelLearning
from stomp.steps.option_learning import OptionLearning
from stomp.steps.planning import Planning


class STOMP:
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
        use_wandb: bool = False,
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

        if use_wandb:
            self._setup_wandb()
        else:
            self.wandb_run = None

    def _setup_wandb(self):
        self.wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=os.getenv("WANDB_ENTITY", None),
            # Set the wandb project where this run will be logged.
            project=os.getenv("WANDB_PROJECT", None),
            # Set the wandb run name to be displayed in the workspace.
            name=str(uuid.uuid4()),
            # Track hyperparameters and run metadata.
            config={
                "env": self.stomp_foundation.env.name,
                "subgoal_state_idx": self.subagoals_state_idx,
                "gamma": self.stomp_foundation.gamma,
                "alpha": self.option_learning.alpha,
                "alpha_prime": self.option_learning.alpha_prime,
                "lambda": self.option_learning.lambda_,
                "lambda_prime": self.option_learning.lambda_prime,
                "alpha_r_model": self.model_learning.alpha_r,
                "alpha_t_model": self.model_learning.alpha_p,
                "alpha_step_size_planning": self.planning.alpha_step_size,
            },
        )
        self.stomp_foundation.wandb_run = self.wandb_run

    def execute(
        self,
        off_policy_steps: int = 50_000,
        num_lookahead_operations: int = 6_000,
        experiment_folder_prefix: str = "stomp",
    ):
        if self.wandb_run is not None:
            self.wandb_run.config["number_of_steps"] = off_policy_steps
            self.wandb_run.config["planning_lookaheads"] = num_lookahead_operations

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

        # Finish wandb integration
        if self.wandb_run is not None:
            self.wandb_run.finish()

        return option_learning_logs, model_learning_logs, planning_logs
