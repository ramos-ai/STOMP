import numpy as np
from tqdm import tqdm

from common.utils import get_progress_bar
from gridworld.gridworld import GridWorld
from stomp.foundation import STOMPFoundation


class OptionLearning(STOMPFoundation):
    def __init__(
        self,
        env: GridWorld,
        gamma: np.float64 = 0.99,
        alpha: np.float64 = 0.1,
        alpha_prime: np.float64 = 0.1,
        lambda_: np.float64 = 0,
        lambda_prime: np.float64 = 0,
    ):
        super().__init__(env, gamma)

        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_options(
        self, hallway_idx: int, off_policy_steps: int = 50_000, log_freq: int = 1_000
    ):
        # Initiating env
        state = self.env.reset()
        state_features = self.env.state_to_features(state)
        initial_state_features = self.env.state_to_features(self.env.initial_state)

        # Lists to store initial state estimative
        initial_state_estimative = []

        progress_bar = (
            range(off_policy_steps)
            if log_freq is None
            else get_progress_bar(off_policy_steps)
        )

        for step in progress_bar:
            initial_state_estimative.append(
                self.linear_combination(
                    initial_state_features, self.w_options[hallway_idx]
                )
            )

            # Chose and execute an action from the equiprobable policy
            action = np.random.choice(self.action_dim, p=self.behavior_policy_probs)
            next_state, reward, done = self.env.step(action)
            next_state_features = self.env.state_to_features(next_state)

            # Calculating the stopping value and checking if the option needs to stop or not
            stopping_value = self.get_stopping_value(next_state_features, hallway_idx)
            stopping = self.should_stop(
                next_state_features, hallway_idx, stopping_value
            )

            option_probs = self.option_policy(state, hallway_idx)
            rho = option_probs[action] / self.behavior_policy_probs[action]

            # Learning Option weights

            delta = self.td_error(
                reward,
                stopping_value,
                self.linear_combination(state_features, self.w_options[hallway_idx]),
                self.linear_combination(
                    next_state_features, self.w_options[hallway_idx]
                ),
                int(stopping),
            )

            # Update option value function
            self.w_options[hallway_idx], self.e_options[hallway_idx] = self.UWT(
                self.w_options[hallway_idx],
                self.e_options[hallway_idx],
                state_features,
                self.alpha * delta,
                rho,
                self.gamma * self.lambda_ * (1 - int(stopping)),
            )

            # Update option policy
            state_action_features = self.env.state_action_to_features(state, action)
            self.theta_options[hallway_idx], self.e_policies[hallway_idx] = self.UWT(
                self.theta_options[hallway_idx],
                self.e_policies[hallway_idx],
                state_action_features,
                self.alpha_prime * delta,
                rho,
                self.gamma * self.lambda_prime * (1 - int(stopping)),
            )

            # Go to next state
            state = next_state
            state_features = next_state_features

            # If we reach the goal, then we reset the state and the eligibility traces
            if done:
                state = self.env.reset()
                state_features = self.env.state_to_features(state)
                self.e_options[hallway_idx] = np.zeros(self.state_dim)
                self.e_policies[hallway_idx] = np.zeros(
                    self.state_dim * self.action_dim
                )

            if log_freq is not None and step % log_freq == 0:
                tqdm.write(
                    f"Step {step}: Option: {hallway_idx}, w_option sum: {self.w_options[hallway_idx].sum()}, v_hat^h(s0): {initial_state_estimative[step]}."
                )

        return initial_state_estimative
