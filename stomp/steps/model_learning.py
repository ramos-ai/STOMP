import numpy as np
from tqdm import tqdm

from common.utils import get_progress_bar
from gridworld.gridworld import GridWorld
from stomp.foundation import STOMPFoundation


class ModelLearning(STOMPFoundation):
    def __init__(
        self,
        env: GridWorld,
        gamma: np.float64 = 0.99,
        alpha_r: np.float64 = 0.1,
        alpha_p: np.float64 = 0.1,
        lambda_: np.float64 = 0,
        lambda_prime: np.float64 = 0,
    ):
        super().__init__(env, gamma)
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_models(
        self, option_idx: int, off_policy_steps: int = 50_000, log_freq: int = 1_000
    ):
        # We need to learn models for all primitive actions and for the full options
        # When we identify that we are dealing with a primitive action,
        # we need to handle the stopping_function and the stopping value accordingly
        is_primitive_action = option_idx < self.action_dim
        hallway_idx = option_idx - self.action_dim

        # Initiating env
        state = self.env.reset()
        state_features = self.env.state_to_features(state)

        # Lists to store the model errors
        reward_model_errors = []
        transition_model_errors = []

        progress_bar = (
            range(off_policy_steps)
            if log_freq is None
            else get_progress_bar(off_policy_steps)
        )

        for step in progress_bar:
            # Chose and execute an action from the equiprobable policy
            action = np.random.choice(self.action_dim, p=self.behavior_policy_probs)
            next_state, reward, done = self.env.step(action)
            next_state_features = self.env.state_to_features(next_state)

            # Check the predicted values from the linear models
            predicted_reward = self.w_rewards[option_idx] @ state_features
            reward_model_error = predicted_reward - reward
            reward_model_errors.append(reward_model_error)

            predicted_transition = self.W_transitions[option_idx] @ state_features
            transition_model_error = np.linalg.norm(
                predicted_transition - next_state_features
            )
            transition_model_errors.append(transition_model_error)

            # Calculating the stopping value and checking if the option needs to stop or not
            stopping_value = (
                None
                if is_primitive_action
                else self.get_stopping_value(next_state_features, hallway_idx)
            )
            stopping = self.should_stop(
                next_state_features,
                hallway_idx,
                stopping_value,
                is_primitive_action,
            )

            # For the case of a primitive action, the probability to take such action is 1
            option_probs = (
                np.ones(self.action_dim)
                if is_primitive_action
                else self.option_policy(state, hallway_idx)
            )
            rho = option_probs[action] / self.behavior_policy_probs[action]

            # Update reward model
            delta_r = self.td_error(
                reward,
                0,
                self.w_rewards[option_idx] @ state_features,
                self.w_rewards[option_idx] @ next_state_features,
                int(stopping),
            )

            self.w_rewards[option_idx], self.e_rewards[option_idx] = self.UWT(
                self.w_rewards[option_idx],
                self.e_rewards[option_idx],
                state_features,
                self.alpha_r * delta_r,
                rho,
                self.gamma * self.lambda_ * (1 - int(stopping)),
            )

            # Update transition model
            for j in range(self.state_dim):
                delta_n = self.td_error(
                    0,
                    next_state_features[j],
                    self.W_transitions[option_idx][j] @ state_features,
                    self.W_transitions[option_idx][j] @ next_state_features,
                    int(stopping),
                )

                (
                    self.W_transitions[option_idx][j],
                    self.e_transitions[option_idx][j],
                ) = self.UWT(
                    self.W_transitions[option_idx][j],
                    self.e_transitions[option_idx][j],
                    state_features,
                    self.alpha_p * delta_n,
                    rho,
                    self.gamma * self.lambda_ * (1 - int(stopping)),
                )

            # Go to next state
            state = next_state
            state_features = next_state_features

            # If we reach the goal, then we reset the state and the eligibility traces
            if done:
                state = self.env.reset()
                state_features = self.env.state_to_features(state)
                self.e_rewards[option_idx] = np.zeros(self.state_dim)
                self.e_transitions[option_idx] = np.zeros(
                    (self.state_dim, self.state_dim)
                )

            if log_freq is not None and step % log_freq == 0:
                tqdm.write(
                    f"Step {step}: Option: {option_idx}, w_reward sum: {self.w_rewards[option_idx].sum()}, W_transition sum: {self.W_transitions[option_idx].sum()}, reward model error: {reward_model_error}, transition model error: {transition_model_error}."
                )

        return reward_model_errors, transition_model_errors


# keeping the original class for now

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from common.option import get_true_option_reward_model, get_true_option_transition_model
from common.statistics import rmse

from gridworld.gridworld import Actions
from stomp.foundation import TemporaryFoundation as Foundation


class TemporaryModelLearning:
    def __init__(
        self,
        foundation: Foundation,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        lambda_: float = 0,
        lambda_prime: float = 0,
    ):
        self.foundation = foundation
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_model(
        self, option_idx: int, off_policy_steps: int = 50_000
    ) -> Tuple[List[float], list[float]]:

        # We need to learn models for all primitive actions and for the full options.
        # In the primitive action case, stopping_function and the stopping value have a different treatment.
        is_primitive_action = option_idx < self.foundation.env.num_actions
        subgoal_idx = option_idx - self.foundation.env.num_actions

        # Get true reward and transition models for statistics
        TRUE_REWARD_MODEL = get_true_option_reward_model(
            self.foundation.env,
            self.foundation,
            option_idx,
            gamma=self.foundation.gamma,
            is_primitive_action=is_primitive_action,
        )
        TRUE_TRANSITION_MODEL = get_true_option_transition_model(
            self.foundation.env,
            self.foundation,
            option_idx,
            gamma=self.foundation.gamma,
            is_primitive_action=is_primitive_action,
        )

        # Initiating env
        done = False
        state = self.foundation.env.reset()

        # Store model errors
        reward_model_rmses = []
        transition_model_errors = []

        for step in tqdm(range(off_policy_steps)):

            # Check goal reached
            if done:
                # Reset
                done = False
                should_stop = False
                state = self.foundation.env.reset()

                # Reset eligibility
                self.foundation.e_rewards[subgoal_idx] = np.zeros(
                    self.foundation.env.num_states
                )
                self.foundation.e_transitions[subgoal_idx] = np.zeros(
                    (self.foundation.env.num_states, self.foundation.env.num_states)
                )

            # Chose and execute an action from the equiprobable policy
            a = np.random.choice(self.foundation.env.num_actions, p=self.foundation.behavior_policy_probs)
            action = Actions(a)

            # Take action
            next_state, reward, done = self.foundation.env.step(action)

            # Get the state features
            state_features = self.foundation.env.get_one_hot_state(state)
            next_state_features = (
                np.zeros_like(state_features)
                if done
                else self.foundation.env.get_one_hot_state(next_state)
            )

            # Store prediciton errors
            reward_model_rmses.append(rmse(
                list(TRUE_REWARD_MODEL.values()),
                [self.foundation.w_rewards[option_idx] @ \
                 self.foundation.env.get_one_hot_state( self.foundation.env.state_idx_to_coordinates[s] )
                 for s in range(self.foundation.env.num_states)]
            ))

            transition_model_errors.append(np.mean(
                [rmse(
                    TRUE_TRANSITION_MODEL[s], self.foundation.W_transitions[option_idx] @ \
                        self.foundation.env.get_one_hot_state( self.foundation.env.state_idx_to_coordinates[s] )
                ) for s in TRUE_TRANSITION_MODEL.keys()]
            ))

            # Calculating the importance sampling ratio for off-policy learning
            if is_primitive_action:
                option_probs = np.zeros(self.foundation.env.num_actions)
                option_probs[action] = 1.  # for primitive actions, the probability is 1. the direction of the action
            else:
                option_probs = self.foundation.softmax_option_policy(state, subgoal_idx)
            importance_sampling_ratio = (option_probs[action] / self.foundation.behavior_policy_probs[action])

            # Handling the stopping value and option probabilities
            if is_primitive_action or done:
                stopping_value = 0
                should_stop = True
            else:
                # Calculating the stopping value and checking if the option needs to stop or not
                stopping_value = self.foundation.get_stopping_value(next_state_features, subgoal_idx)
                should_stop = self.foundation.should_stop(next_state_features, subgoal_idx, stopping_value)

            # Learning the Reward model
            # TD Error for reward model
            delta_r = self.foundation.td_error(
                reward,
                0,
                float(self.foundation.w_rewards[option_idx] @ state_features),
                float(self.foundation.w_rewards[option_idx] @ next_state_features),
                should_stop,
            )

            # Learning Reward Model Weights
            (
                self.foundation.w_rewards[option_idx],
                self.foundation.e_rewards[option_idx],
            ) = self.foundation.UWT(
                self.foundation.w_rewards[option_idx],
                self.foundation.e_rewards[option_idx],
                state_features,
                self.alpha_r * delta_r,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_ * (1 - should_stop),
            )

            for j in range(self.foundation.env.num_states):
                delta_n = self.foundation.td_error(
                    0,
                    next_state_features[j],
                    self.foundation.W_transitions[option_idx][j] @ state_features,
                    self.foundation.W_transitions[option_idx][j] @ next_state_features,
                    int(should_stop),
                )
                (
                    self.foundation.W_transitions[option_idx][j],
                    self.foundation.e_transitions[option_idx][j],
                ) = self.foundation.UWT(
                    self.foundation.W_transitions[option_idx][j],
                    self.foundation.e_transitions[option_idx][j],
                    state_features,
                    self.alpha_p * delta_n,
                    importance_sampling_ratio,
                    self.foundation.gamma * self.lambda_ * (1 - int(should_stop)),
                )

            # Learning the Transition model
            # In the original paper, the transition model is learned for each state
            # We introduce a vectorized version of the transition model learning
            # predicted_state_feature = (
            #     self.foundation.W_transitions[option_idx] @ state_features
            # )
            # predicted_next_state_feature = (
            #     self.foundation.W_transitions[option_idx] @ next_state_features
            # )
            # delta_vec = self.foundation.td_error(
            #     0,
            #     next_state_features,
            #     predicted_state_feature,
            #     predicted_next_state_feature,
            #     should_stop,
            # )

            # (
            #     self.foundation.W_transitions[option_idx],
            #     self.foundation.e_transitions[option_idx],
            # ) = self.foundation.vecUWT(
            #     self.foundation.W_transitions[option_idx],
            #     self.foundation.e_transitions[option_idx],
            #     state_features,
            #     self.alpha_p * delta_vec,
            #     importance_sampling_ratio,
            #     self.foundation.gamma * self.lambda_ * (1 - should_stop),
            # )

            # Moving to the next state
            state = next_state
            state_features = next_state_features

        return reward_model_rmses, transition_model_errors
