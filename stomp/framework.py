import os

import numpy as np
from tqdm import tqdm

from common.utils import save_learning_results
from gridworld.gridworld import GridWorld, State
from stomp.steps.model_learning import ModelLearning
from stomp.steps.option_learning import OptionLearning
from stomp.steps.planning import Planning


class STOMP(OptionLearning, ModelLearning, Planning):
    def __init__(
        self,
        env: GridWorld,
        gamma: np.float64 = 0.99,
        alpha: np.float64 = 0.1,
        alpha_prime: np.float64 = 0.1,
        alpha_r: np.float64 = 0.1,
        alpha_p: np.float64 = 0.1,
        lambda_: np.float64 = 0,
        lambda_prime: np.float64 = 0,
        alpha_step_size: np.float64 = 1.0,
        experiment_name: str = "reward_respecting_options_and_primitive_actions",
    ):
        self.experiment_name = experiment_name
        OptionLearning.__init__(
            self=self,
            env=env,
            gamma=gamma,
            alpha=alpha,
            alpha_prime=alpha_prime,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        ModelLearning.__init__(
            self=self,
            env=env,
            gamma=gamma,
            alpha_r=alpha_r,
            alpha_p=alpha_p,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        Planning.__init__(
            self=self, env=env, gamma=gamma, alpha_step_size=alpha_step_size
        )

    def execute_option_learning(
        self, off_policy_steps: int = 50_000, log_freq: int = 1_000
    ):
        option_learning = []
        for hallway_idx in range(self.num_hallways):
            tqdm.write(
                f"\nLearning options for hallway {hallway_idx} with {off_policy_steps} off-policy steps...\n"
            )
            initial_state_estimative = self.learn_options(
                hallway_idx=hallway_idx,
                off_policy_steps=off_policy_steps,
                log_freq=log_freq,
            )
            option_learning.append(initial_state_estimative)
            save_learning_results(
                initial_state_estimative,
                os.path.join(self.experiment_name, "option_learning"),
                f"initial_state_estimative_hallway_{hallway_idx}.pkl",
            )
            tqdm.write(f"Logs saved for hallway {hallway_idx}.\n")
        return option_learning

    def execute_models_option_learning(
        self, off_policy_steps: int = 50_000, log_freq: int = 1_000
    ):
        option_model_learning = []
        for option_idx in range(self.num_options):
            tqdm.write(
                f"\nLearning models for Option {option_idx}, {'a Primitive Action' if option_idx < self.action_dim else 'a Full Option'}, with {off_policy_steps} off-policy steps...\n"
            )
            reward_model_errors, transition_model_errors = self.learn_models(
                option_idx=option_idx,
                off_policy_steps=off_policy_steps,
                log_freq=log_freq,
            )
            option_model_learning.append((reward_model_errors, transition_model_errors))
            save_learning_results(
                reward_model_errors,
                os.path.join(self.experiment_name, "model_learning"),
                f"reward_model_errors_{option_idx}.pkl",
            )
            save_learning_results(
                transition_model_errors,
                os.path.join(self.experiment_name, "model_learning"),
                f"transition_model_errors_{option_idx}.pkl",
            )
            tqdm.write(f"Logs saved for option {option_idx}.\n")
        return option_model_learning

    def execute_planning_with_options(
        self, lookahead_operations: int = 6_000, log_freq: int = 100
    ):
        tqdm.write(
            f"\nPlanning with the learned options with {lookahead_operations} operations steps...\n"
        )
        initial_state_planning_estimative = self.plan_with_options(
            num_lookahead_operations=lookahead_operations, log_freq=log_freq
        )
        save_learning_results(
            initial_state_planning_estimative,
            os.path.join(self.experiment_name, "planning_with_options"),
            "initial_state_planning_estimative.pkl",
        )
        return initial_state_planning_estimative

    def learn(
        self,
        off_policy_steps: int = 50_000,
        lookahead_operations: int = 6_000,
        log_freq: int = 1_000,
    ):
        """
        Learn the STOMP algorithm.
        """

        option_learning_logs = self.execute_option_learning(
            off_policy_steps=off_policy_steps, log_freq=log_freq
        )

        assert self.w_options[0].sum() != 0, (
            "The option value function is not updated. Check the learning process."
        )

        save_learning_results(
            self.w_options,
            os.path.join(self.experiment_name, "option_learning"),
            "w_options.pkl",
        )

        tqdm.write("\nw_options saved!")

        options_model_learning_logs = self.execute_models_option_learning(
            off_policy_steps=off_policy_steps, log_freq=log_freq
        )

        assert self.w_rewards[0].sum() != 0, (
            "The reward model is not updated. Check the learning process."
        )

        save_learning_results(
            self.w_rewards,
            os.path.join(self.experiment_name, "model_learning"),
            "w_rewards.pkl",
        )

        save_learning_results(
            self.W_transitions,
            os.path.join(self.experiment_name, "model_learning"),
            "W_transitions.pkl",
        )

        tqdm.write("\w_rewards and W_transitions saved!")

        planning_logs = self.execute_planning_with_options(
            lookahead_operations=lookahead_operations, log_freq=log_freq
        )

        assert self.w.sum() != 0, (
            "The planning value function is not updated. Check the learning process."
        )

        save_learning_results(
            self.w,
            os.path.join(self.experiment_name, "planning_with_options"),
            "w.pkl",
        )

        tqdm.write("\w saved!")

        tqdm.write("\nLearning Finished!")

        return option_learning_logs, options_model_learning_logs, planning_logs
