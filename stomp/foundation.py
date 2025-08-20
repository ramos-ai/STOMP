from os.path import join
from typing import List

import numba
import numpy as np
from numpy.typing import NDArray
import wandb

from gridworld.gridworld import Actions, GridWorld, State


@numba.jit(nopython=True)
def _td_error_compute(
    cumulant: float,
    stopping_value: float,
    current_value: float,
    next_value: float,
    stopping_prob: bool,
    gamma: float,
) -> float:
    return (
        cumulant
        + int(stopping_prob) * stopping_value
        + gamma * next_value * (1 - int(stopping_prob))
        - current_value
    )


@numba.jit(nopython=True)
def _uwt_compute(
    w: NDArray,
    e: NDArray,
    gradient: NDArray,
    alpha_delta: float,
    rho: float,
    gamma_lambda: float,
) -> tuple[NDArray, NDArray]:
    e = rho * (e + gradient)
    w = w + alpha_delta * e
    e = gamma_lambda * e
    return w, e


@numba.jit(nopython=True)
def _vec_uwt_compute(
    w: NDArray,
    e: NDArray,
    gradient: NDArray,
    alpha_delta_vec: NDArray,
    rho: float,
    gamma_lambda: float,
) -> tuple[NDArray, NDArray]:
    e = rho * (e + gradient)
    w = w + alpha_delta_vec[:, np.newaxis] * e
    # w = w + np.einsum("i,ij->ij", alpha_delta_vec, e)
    # w = w + alpha_delta_vec.reshape(-1, 1) * e
    e = gamma_lambda * e
    return w, e


class Foundation:
    def __init__(
        self,
        env: GridWorld,
        subgoals_state: List[State],
        subgoals_state_idx: List[int],
        behavior_policy_probs: NDArray | None = None,
        gamma: float = 0.99,
    ):
        self.env = env
        self.gamma = gamma
        self.subgoals_state = subgoals_state
        self.subgoals_state_idx = subgoals_state_idx
        self.num_subgoals = len(self.subgoals_state_idx)
        self.num_options = len(Actions) + self.num_subgoals
        self.behavior_policy_probs = behavior_policy_probs
        if self.behavior_policy_probs is None:
            self.behavior_policy_probs = (
                np.ones(self.env.num_actions) / self.env.num_actions
            )

        # Setup WandB logging
        # wandb is set in STOMP class (_setup_wandb)
        self.wandb_run: wandb.Run | None = None

        # For STOMP we don't consider the GOAL state, while for Successor we do
        # Therefore, when we change context from Successor to STOMP we need to reset the environment
        # And vice-versa
        self.env.reset()

        ################## STOMP Step 2 Parameters: Options Learning ##################

        # Option parameters (for each subgoal)
        self.w_bonus = 1.0  # Bonus weight for subgoal features

        # Option value functions and policies (one for each subgoal)
        self.w_subgoal = [
            np.zeros(self.env.num_states) for _ in range(self.num_subgoals)
        ]
        self.theta_subgoal = [
            np.zeros(self.env.num_states * self.env.num_actions)
            for _ in range(self.num_subgoals)
        ]

        # Eligibility traces (one for each subgoal)
        self.e_options = [
            np.zeros(self.env.num_states) for _ in range(self.num_subgoals)
        ]
        self.e_policies = [
            np.zeros(self.env.num_states * self.env.num_actions)
            for _ in range(self.num_subgoals)
        ]

        ################## STOMP Step 3 Parameters: Model Learning ##################

        # Models for rewards and transitions (one for each OPTION)
        self.w_rewards = [
            np.zeros(self.env.num_states) for _ in range(self.num_options)
        ]
        self.W_transitions = [
            np.zeros((self.env.num_states, self.env.num_states))
            for _ in range(self.num_options)
        ]

        # Model eligibility traces (one for each OPTION)
        self.e_rewards = [
            np.zeros(self.env.num_states) for _ in range(self.num_options)
        ]
        self.e_transitions = [
            np.zeros((self.env.num_states, self.env.num_states))
            for _ in range(self.num_options)
        ]

        ################## STOMP Step 4 Parameters: Planning with Options ##################

        # Main task value function
        self.w = np.zeros(self.env.num_states)

    def save_vectors(self, base_path: str) -> None:
        np.save(join(f"{base_path}", "w_subgoal.npy"), self.w_subgoal)
        np.save(join(f"{base_path}", "theta_subgoal.npy"), self.theta_subgoal)
        np.save(join(f"{base_path}", "w_rewards.npy"), self.w_rewards)
        np.save(join(f"{base_path}", "W_transitions.npy"), self.W_transitions)
        np.save(join(f"{base_path}", "w.npy"), self.w)

    # WARNING: subgoal_idx != subgoal_state_idx
    # subgoal_idx refers to which subgoal is been processed
    # subgoal_state_idx refers to the idx of the subgoal state in the grid room
    def softmax_option_policy(self, state: State, subgoal_idx: int) -> NDArray:
        state_action_values = np.zeros(self.env.num_actions)

        for a in range(self.env.num_actions):
            action = Actions(a)
            features = self.env.get_one_hot_state_action(state, action)
            state_action_values[action] = features @ self.theta_subgoal[subgoal_idx]

        exp_values = np.exp(state_action_values)
        probs = exp_values / np.sum(exp_values)
        return probs

    # In the original paper, this is the z function
    def get_stopping_value(self, state_features: NDArray, subgoal_idx: int) -> float:
        subgoal_state_idx = self.subgoals_state_idx[subgoal_idx]
        hallway_feature = state_features[subgoal_state_idx]
        w_feature = self.w[subgoal_state_idx]
        return (
            self.w @ state_features
            - w_feature * hallway_feature
            + self.w_bonus * hallway_feature
        )

    # In the original paper, this is the beta function from the option
    def should_stop(
        self,
        state_features: NDArray,
        subgoal_idx: int,
        stopping_value: float | None = None,
        is_primitive_action: bool = False,
    ) -> bool:
        if is_primitive_action:
            return True
        if stopping_value is None:
            stopping_value = self.get_stopping_value(state_features, subgoal_idx)
        option_value = self.w_subgoal[subgoal_idx] @ state_features
        return bool(stopping_value >= option_value)

    def td_error(
        self,
        cumulant: float | NDArray[np.floating],
        stopping_value: float | NDArray[np.floating],
        current_value: float | NDArray[np.floating],
        next_value: float | NDArray[np.floating],
        stopping_prob: bool | int,
    ) -> float:
        return _td_error_compute(
            cumulant,
            stopping_value,
            current_value,
            next_value,
            stopping_prob,
            self.gamma,
        )

    def UWT(
        self,
        w: NDArray,
        e: NDArray,
        gradient: NDArray,
        alpha_delta: float,
        rho: float,
        gamma_lambda: float,
    ) -> tuple[NDArray, NDArray]:
        return _uwt_compute(w, e, gradient, alpha_delta, rho, gamma_lambda)

    def vecUWT(
        self,
        w: NDArray,
        e: NDArray,
        gradient: NDArray,
        alpha_delta_vec: NDArray,
        rho: float,
        gamma_lambda: float,
    ) -> tuple[NDArray, NDArray]:
        return _vec_uwt_compute(w, e, gradient, alpha_delta_vec, rho, gamma_lambda)
