import numpy as np

from gridworld.gridworld import GridWorld, State


def value_iteration(
        env: GridWorld,
        gamma: float=0.99,
        threshold: float=0.01,
        bonus_state: State=None,
        reward_bonus: float=1.
        ) -> np.ndarray:
    """ Implement value iteration algorithm to find the optimal value function. """

    # Initialize the value function
    V = np.zeros(env.num_states)

    while True:
        delta = 0
        for s in range(env.num_states):
            v = V[s]

            # Store value of each action
            action_values = np.zeros(env.num_actions)

            # Calculate the value for each action
            for a in range(env.num_actions):
                env.reset()
                env.current_state = env.state_idx_to_coordinates[s]
                s_, r, done = env.step(a)

                # Add bonus for reaching specified state
                if s_ == bonus_state:
                    r += reward_bonus
                    done = True

                if done:
                    action_values[a] = r
                else:
                    next_state = env.state_coordinates_to_idx[s_]
                    action_values[a] = r + gamma * V[next_state]
            V[s] = max(action_values)

            delta = max(delta, abs(v - V[s]))
        if delta < threshold:
            break

    env.reset()
    return V


if __name__ == "__main__":
    import argparse

    from gridworld.gridworld import GridWorld
    from gridworld.room_design import stomp_two_room_design as room_design

    parser = argparse.ArgumentParser(description="Value Iteration Learning")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--threshold", type=float, default=0.01, help="Convergence threshold")
    parser.add_argument("--bonus_state", type=State, default=(7,3), help="State to receive bonus reward")
    parser.add_argument("--reward_bonus", type=float, default=1.0, help="Bonus reward for reaching the bonus state")
    parser.add_argument("--log", action="store_true", help="Log the value function")

    # Parse command line arguments
    args = parser.parse_args()

    gamma = args.gamma
    threshold = args.threshold
    bonus_state = args.bonus_state
    reward_bonus = args.reward_bonus

    # Initialize the environment
    env = GridWorld(room_design)

    # Get the optimal policy using approximate value function learning
    V = value_iteration(
        env=env,
        gamma=gamma,
        threshold=threshold,
        bonus_state=bonus_state,
        reward_bonus=reward_bonus
    )

    import math
    # Check if the learned option value function is close to the expected value
    assert math.isclose(V[24], 0.895, rel_tol=0.05)

    if args.log:
        while True:
            s = int(input('State (int): '))
            if s < 0 or s >= env.num_states:
                print("Invalid state index")
                continue
            print(f"Value for state {s}: {V[s]}")
