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

    def linear_combination(self, state_features: NDArray, weights: NDArray) -> float:
        """Calculate linear combination of features and weights"""
        return np.dot(state_features, weights)

    def option_policy(self, state: State, hallway_idx: np.int64) -> NDArray:
        """Softmax policy for a specific hallway option"""
        state_action_values = np.zeros(self.action_dim)
        for a in range(self.action_dim):
            features = self.env.state_action_to_features(state, a)
            state_action_values[a] = self.linear_combination(
                features, self.theta_options[hallway_idx]
            )

        exp_values = np.exp(state_action_values)
        probs = exp_values / np.sum(exp_values)
        return probs

    def get_stopping_value(
        self, state_features: NDArray, hallway_idx: np.int64
    ) -> np.float64:
        """Calculate stopping value for a specific hallway option"""
        _, _, hallway_state_feature_idx = self.hallways_info[hallway_idx]
        hallway_feature = state_features[hallway_state_feature_idx]
        w_feature = self.w[hallway_state_feature_idx]
        return (
            self.linear_combination(self.w, state_features)
            - w_feature * hallway_feature
            + self.w_bonus * hallway_feature
        )

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
        option_value = self.linear_combination(
            state_features, self.w_options[hallway_idx]
        )
        return stopping_value >= option_value

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
