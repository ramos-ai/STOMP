import numpy as np

from gridworld.gridworld import GridWorld
from stomp.foundation import Foundation


def get_true_option_reward_model(
        env: GridWorld,
        # task: Subtask,
        # option: Option,
        foundation: Foundation,
        option_idx: int,
        gamma: float =0.99,
        is_primitive_action: bool = False):
    """ Get the true reward model for the option in the environment for a given task. Eq (12) in the paper. """

    TRUE_REWARD_MODEL = {s: 0. for s in range(env.num_states)}
    for s in range(env.num_states):

        # Reset environment to current state
        _ = env.reset()
        state = env.state_idx_to_coordinates[s]
        env.current_state = state
        stop = False

        # Execute option
        rewards = []
        while not stop:
            # Sample next state using the option model
                # a = option.choose_action(env, env.state)
            if is_primitive_action:
                a = option_idx
            else:
                subgoal_idx = option_idx - foundation.env.num_actions
                probs = foundation.softmax_option_policy(env.current_state, subgoal_idx)
                a = np.argmax(probs)
            next_state, reward, done = env.step(a)

            # Store reward
            rewards.append(reward)

            # Check if option should stop
                # stop = task.B(next_state, option.w)
            if done or is_primitive_action:
                stop = True
                continue

            next_state_features = env.get_one_hot_state(next_state)
            stop = foundation.should_stop(next_state_features, subgoal_idx)

        # Compute state return
        G = sum( [gamma**(t-1) * r for t, r in enumerate(rewards, start=1) ] )
        TRUE_REWARD_MODEL[s] = G

    return TRUE_REWARD_MODEL


def get_true_option_transition_model(
        env: GridWorld,
        # task: Subtask,
        # option: Option,
        foundation: Foundation,
        option_idx: int,
        gamma: float =0.99,
        is_primitive_action: bool = False):
    """ Get the true transition model for the option in the environment for a given task. Eq (13) in the paper."""

    TRUE_TRANSITION_MODEL = {s: [0. for _ in range(env.num_states)] for s in range(env.num_states)}
    for s in range(env.num_states):

        # Reset environment to current state
        _ = env.reset()
        state = env.state_idx_to_coordinates[s]
        env.current_state = state
        stop = False

        # Execute option
        t = 0
        while not stop:
            # Sample next state using the option model
                # a = option.choose_action(env, env.state)
            if is_primitive_action:
                a = option_idx
            else:
                subgoal_idx = option_idx - foundation.env.num_actions
                probs = foundation.softmax_option_policy(env.current_state, subgoal_idx)
                a = np.argmax(probs)
            next_state, reward, done = env.step(a)

            # Increase t
            t += 1

            # Check if option should stop
                # stop = task.B(next_state, option.w)
            if done or is_primitive_action:
                stop = True
                continue

            next_state_features = env.get_one_hot_state(next_state)
            stop = foundation.should_stop(next_state_features, subgoal_idx)

        # Get true next state features
        if done:
            next_state_features = np.zeros(env.num_states)  # Terminal state
        else:
            next_state_features = env.get_one_hot_state(next_state)
        Pr = (gamma**t) * next_state_features
        TRUE_TRANSITION_MODEL[s] = Pr

    return TRUE_TRANSITION_MODEL
