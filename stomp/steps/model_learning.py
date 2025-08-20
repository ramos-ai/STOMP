from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from common.option import get_true_option_reward_model, get_true_option_transition_model
from common.statistics import rmse
from gridworld.gridworld import Actions
from stomp.foundation import Foundation


class ModelLearning:
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
