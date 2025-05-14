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

            # Update models weights

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
