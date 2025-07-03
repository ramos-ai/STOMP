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
                self.w_options[hallway_idx] @ initial_state_features
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
                self.w_options[hallway_idx] @ state_features,
                self.w_options[hallway_idx] @ next_state_features,
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


# keeping the original class for now

from typing import List

import numpy as np
from tqdm import tqdm

from common.statistics import rmse
from gridworld.gridworld import Actions
from stomp.foundation import TemporaryFoundation as Foundation


class TemporaryOptionLearning:
    def __init__(
        self,
        foundation: Foundation,
        alpha: float = 0.1,
        alpha_prime: float = 0.1,
        lambda_: float = 0,
        lambda_prime: float = 0,
    ):
        self.foundation = foundation
        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_options(
        self, subgoal_idx: int, off_policy_steps: int = 50_000, return_rmse: bool = False
    ) -> List[float]:

        # Initiating env
        done = False
        state = self.foundation.env.reset()
        state_features = self.foundation.env.get_one_hot_state(state)
        initial_state_features = state_features

        initial_state_estimative = []
        # To calculate the RMSE of the states we need the optimal value function
        if return_rmse:
            from common.rl import value_iteration
            OPTIMAL_V = value_iteration(
                self.foundation.env,
                bonus_state=self.foundation.subgoals_state[subgoal_idx]
            )
            rmse_of_state_values = []

        for step in tqdm(range(off_policy_steps)):

            # Check goal reached
            if done:

                # Reset
                done = False
                should_stop = False
                state = self.foundation.env.reset()

                # Reset eligibility traces
                self.foundation.e_options[subgoal_idx] = np.zeros(
                    self.foundation.env.num_states
                )
                self.foundation.e_policies[subgoal_idx] = np.zeros(
                    self.foundation.env.num_states * self.foundation.env.num_actions
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

            # Store statistics
            initial_state_estimative.append(self.foundation.w_subgoal[subgoal_idx] @ initial_state_features)
            if return_rmse:
                rmse_of_state_values.append(
                    rmse(
                        [OPTIMAL_V[s] for s in range(self.foundation.env.num_states)],
                        [self.foundation.w_subgoal[subgoal_idx] @ \
                         self.foundation.env.get_one_hot_state( self.foundation.env.state_idx_to_coordinates[s] )
                         for s in range(self.foundation.env.num_states)]
                    )
                )

            # Calculating the importance sampling ratio for off-policy learning
            option_probs = self.foundation.softmax_option_policy(state, subgoal_idx)
            importance_sampling_ratio = (option_probs[action] / self.foundation.behavior_policy_probs[action])

            if done:
                stopping_value = 0
                should_stop = True
            else:
                # Calculating the stopping value and checking if the option needs to stop or not
                stopping_value = self.foundation.get_stopping_value(next_state_features, subgoal_idx)
                should_stop = self.foundation.should_stop(next_state_features, subgoal_idx, stopping_value)

            # Calculating TD Error
            delta = self.foundation.td_error(
                reward,
                stopping_value,
                float(self.foundation.w_subgoal[subgoal_idx] @ state_features),
                float(self.foundation.w_subgoal[subgoal_idx] @ next_state_features),
                should_stop,
            )

            # Learning Option Weights
            (
                self.foundation.w_subgoal[subgoal_idx],
                self.foundation.e_options[subgoal_idx],
            ) = self.foundation.UWT(
                self.foundation.w_subgoal[subgoal_idx],
                self.foundation.e_options[subgoal_idx],
                state_features,
                self.alpha * delta,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_ * (1 - should_stop),
            )

            # Learning Option Policy
            state_action_features = self.foundation.env.get_one_hot_state_action(
                state, action
            )
            (
                self.foundation.theta_subgoal[subgoal_idx],
                self.foundation.e_policies[subgoal_idx],
            ) = self.foundation.UWT(
                self.foundation.theta_subgoal[subgoal_idx],
                self.foundation.e_policies[subgoal_idx],
                state_action_features,
                self.alpha_prime * delta,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_prime * (1 - should_stop),
            )

            # Moving to the next state
            state = next_state
            state_features = next_state_features

        if return_rmse:
            return initial_state_estimative, rmse_of_state_values
        else:
            return initial_state_estimative


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Learning TwoRooms")

    parser.add_argument("--rep", type=int, default=1, help="Replications")
    parser.add_argument("--number_of_steps", type=int, default=50000, help="Number of steps")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.0, help="Trace decay parameter")
    parser.add_argument("--alpha_", type=float, default=0.1, help="Learning rate for actor")
    parser.add_argument("--lmbda_", type=float, default=0.0, help="Trace decay parameter for actor")

    # Parse the arguments
    args = parser.parse_args()

    rep = args.rep
    number_of_steps = args.number_of_steps
    alpha = args.alpha
    gamma = args.gamma
    lmbda = args.lmbda
    alpha_ = args.alpha_
    lmbda_ = args.lmbda_

    number_of_steps = 50_000
    subgoal_state_idx = 30

    # STOMP setup
    from gridworld.gridworld import TemporaryGridWorld as GridWorld
    from gridworld.room_design import two_room_design as room_design
    from stomp.framework import STOMP

    env = GridWorld(
        room_array=room_design, success_prob=1.0
    )

    # Statistics
    many_initial_state_estimatives = np.zeros((rep, number_of_steps))
    many_rmse_of_states = np.zeros((rep, number_of_steps))

    for i in range(rep):
        print(f"Replication {i + 1}/{rep}")

        # TODO: Reset option learning parameters w and theta
        env.reset()
        stomp = STOMP(
            env=env,
            subgoal_states_info={subgoal_state_idx: (7, 3)},
        )

        # Local access to option learning
        option_learning = stomp.option_learning

        initial_state_estimative, rmse_of_states = option_learning.learn_options(
            0, number_of_steps, return_rmse=True
        )

        # Store results
        many_initial_state_estimatives[i] = initial_state_estimative
        many_rmse_of_states[i] = rmse_of_states


    # Plot the results
    import matplotlib.pyplot as plt

    # Calculate the averages
    average_v_s0 = np.mean(many_initial_state_estimatives, axis=0)
    average_rmse = np.mean(many_rmse_of_states, axis=0)

    stddev_v_s0 = np.std(many_initial_state_estimatives, axis=0)
    stddev_rmse = np.std(many_rmse_of_states, axis=0)

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis (left y-axis) for error (red)
    ax1.set_xlabel('Steps of off-policy experience', fontsize=12, color='black')
    ax1.set_ylabel('Error between $v^0$ and $v^h$', color='coral', fontsize=12)
    ax1.plot(range(number_of_steps), average_rmse, color='coral', linewidth=2)
    ax1.fill_between(
        range(number_of_steps),
        average_rmse - stddev_rmse,
        average_rmse + stddev_rmse,
        color='coral',
        alpha=0.3
    )
    ax1.tick_params(axis='y', labelcolor='coral')
    ax1.set_xlim(0, number_of_steps)
    ax1.set_ylim(0, 2.2)
    ax1.grid(False)

    # Secondary axis (right y-axis) for value function (blue)
    ax2 = ax1.twinx()
    ax2.set_ylabel('$v^h(s_0)$', color='skyblue', fontsize=12)
    ax2.plot(range(number_of_steps), average_v_s0, color='skyblue', linewidth=2)
    ax2.fill_between(
        range(number_of_steps),
        average_v_s0 - stddev_v_s0,
        average_v_s0 + stddev_v_s0,
        color='skyblue',
        alpha=0.3
    )
    ax2.tick_params(axis='y', labelcolor='skyblue')
    ax2.set_ylim(-1.8, 1)
    ax2.grid(False)

    # Add a horizontal dotted line for reference
    ax2.axhline(y=0.895, color='skyblue', linestyle='dotted', alpha=0.7)
    ax2.text(0, 0.83, "$v^h(s_0)$", color='skyblue', fontsize=12)

    # Style the plot
    for spine in ax1.spines.values():
        spine.set_color('black')
    for spine in ax2.spines.values():
        spine.set_color('black')

    ax1.tick_params(colors='black')
    ax2.tick_params(colors='black')
    ax1.xaxis.label.set_color('black')

    plt.tight_layout()
    plt.show()
