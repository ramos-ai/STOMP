import numpy as np
from numpy.typing import NDArray

from gridworld.gridworld import GridWorld, State


class STOMPFoundation:
    def __init__(
        self,
        env: GridWorld,
        gamma: np.float64 = 0.99,
    ):
        self.env = env
        self.gamma = gamma

        # Feature dimensions
        self.state_dim = env.num_states - 1  # Exclude terminal state
        self.action_dim = env.num_actions

        # Get hallway information from environment
        self.hallways_info = env.hallways_info
        self.num_hallways = len(self.hallways_info)

        # Total of options
        self.num_options = self.action_dim + self.num_hallways

        # The policy used to take actions
        # Off-policy learning
        self.behavior_policy_probs = np.ones(self.action_dim) / self.action_dim

        ################################## STOMP Step 2 Parameters: Options Learning ##################################

        # Option parameters (for each hallway)
        self.w_bonus = 1.0  # Bonus weight for hallway features

        # Option value functions and policies (one for each hallway)
        self.w_options = [np.zeros(self.state_dim) for _ in range(self.num_hallways)]
        self.theta_options = [
            np.zeros(self.state_dim * self.action_dim) for _ in range(self.num_hallways)
        ]

        # Eligibility traces (one for each hallway)
        self.e_options = [np.zeros(self.state_dim) for _ in range(self.num_hallways)]
        self.e_policies = [
            np.zeros(self.state_dim * self.action_dim) for _ in range(self.num_hallways)
        ]

        ################################## STOMP Step 3 Parameters: Model Learning ##################################

        # Models for rewards and transitions (one for each OPTION)
        self.w_rewards = [np.zeros(self.state_dim) for _ in range(self.num_options)]
        self.W_transitions = [
            np.zeros((self.state_dim, self.state_dim)) for _ in range(self.num_options)
        ]

        # Model eligibility traces (one for each OPTION)
        self.e_rewards = [np.zeros(self.state_dim) for _ in range(self.num_options)]
        self.e_transitions = [
            np.zeros((self.state_dim, self.state_dim)) for _ in range(self.num_options)
        ]

        ################################## STOMP Step 4 Parameters: Planning with Options ##################################

        # Main task value function
        # self.w = np.random.uniform(-0.1, 0.1, self.state_dim)
        self.w = np.zeros(self.state_dim)

    def option_policy(self, state: State, hallway_idx: np.int64) -> NDArray:
        """Softmax policy for a specific hallway option"""
        state_action_values = np.zeros(self.action_dim)
        for a in range(self.action_dim):
            features = self.env.state_action_to_features(state, a)
            state_action_values[a] = features @ self.theta_options[hallway_idx]

        exp_values = np.exp(state_action_values)
        probs = exp_values / np.sum(exp_values)
        return probs

    # In the original paper, this is the z function
    def get_stopping_value(
        self, state_features: NDArray, hallway_idx: np.int64
    ) -> np.float64:
        """Calculate stopping value for a specific hallway option"""
        _, _, hallway_state_feature_idx = self.hallways_info[hallway_idx]
        hallway_feature = state_features[hallway_state_feature_idx]
        w_feature = self.w[hallway_state_feature_idx]
        return (
            self.w @ state_features
            - w_feature * hallway_feature
            + self.w_bonus * hallway_feature
        )

    # In the original paper, this is the beta function from the option
    def should_stop(
        self,
        state_features: NDArray,
        hallway_idx: np.int64,
        stopping_value: np.float64 = None,
        is_primitive_action: bool = False,
    ) -> bool:
        """Determine if specific hallway option should stop"""
        if is_primitive_action:
            return True
        if stopping_value is None:
            stopping_value = self.get_stopping_value(state_features, hallway_idx)
        option_value = self.w_options[hallway_idx] @ state_features
        return stopping_value >= option_value

    # This compute the general value function (GVF) for the option
    # Using Bellman's equation
    def compute_gvf(self, hallway_idx: int = 0, tol: float = 1e-6, max_iter: int = 500):
        # 1. Build static transition model for this option/policy
        all_states = self.env.get_all_states()
        P = {s: {} for s in all_states}  # P[s][s'] = P^π(s'|s)

        for s in all_states:
            pi_probs = self.option_policy(s, hallway_idx)
            P[s] = {}
            for a, pa in enumerate(pi_probs):
                # query, but do not change env.current_state
                old_state = self.env.current_state
                self.env.current_state = s
                s_next, reward, done = self.env.step(a)
                self.env.current_state = old_state

                P[s].setdefault(s_next, 0.0)
                P[s][s_next] += pa  # accumulate probability mass

        # 2. Initialize v-table
        v = {s: 0.0 for s in all_states}

        # 3. Gauss–Seidel style sweeps
        for _ in range(max_iter):
            delta = 0.0
            for s in all_states:
                v_old = v[s]
                total = 0.0

                for s_next, p_s in P[s].items():
                    # get c, β, z at s_next
                    feat = self.env.state_to_features(s_next)
                    c_t1 = self.env.get_reward(s_next)  # or however you get c(s)
                    beta_t1 = self.should_stop(
                        feat, hallway_idx, self.get_stopping_value(feat, hallway_idx)
                    )
                    z_t1 = self.get_stopping_value(feat, hallway_idx)

                    if beta_t1:
                        # terminal: just reward + γ·z
                        total += p_s * (c_t1 + self.gamma * z_t1)
                    else:
                        cont = (1 - beta_t1) * v[s_next] + beta_t1 * z_t1
                        total += p_s * (c_t1 + self.gamma * cont)

                v[s] = total
                delta = max(delta, abs(v[s] - v_old))

            if delta < tol:
                break

        return v

    def td_error(
        self,
        cumulant: np.float64,
        stopping_value: np.float64,
        current_value: np.float64,
        next_value: np.float64,
        stopping_prob: bool,
    ) -> np.float64:
        return (
            cumulant
            + stopping_prob * stopping_value
            + self.gamma * next_value * (1 - stopping_prob)
            - current_value
        )

    def UWT(
        self,
        w: NDArray,
        e: NDArray,
        gradient: NDArray,
        alpha_delta: np.float64,
        rho: np.float64,
        gamma_lambda: np.float64,
    ) -> tuple[NDArray, NDArray]:
        e = rho * (e + gradient)
        w = w + alpha_delta * e
        e = gamma_lambda * e
        return w, e


# keeping the original class for now

from os.path import join
from typing import List

import numba
import numpy as np
from numpy.typing import NDArray

from gridworld.gridworld import Actions, TemporaryGridWorld as GridWorld, State


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


class TemporaryFoundation:
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

        # For STOMP we don't consider the GOAL state, while for Successor we do
        # Therefore, when we change context from Successor to STOMP we need to reset the environment
        # And vice-versa
        self.env.reset()

        ################################## STOMP Step 2 Parameters: Options Learning ##################################

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

        ################################## STOMP Step 3 Parameters: Model Learning ##################################

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

        ################################## STOMP Step 4 Parameters: Planning with Options ##################################

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
