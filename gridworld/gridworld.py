import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


State = Tuple[NDArray[np.float64], NDArray[np.float64]]
HallwaysInfo = List[Tuple[State, NDArray[np.float64], int]]
StepReturns = Tuple[State, np.int64, bool]


class GridWorld:
    """A GridWorld environment implementation for reinforcement learning tasks.

    This class implements a grid world where an agent can navigate through different types of cells:
    - Free spaces (0)
    - Goal state (1)
    - Penalty areas (-1)
    - Walls (9)
    - Hallways (8)
    - Start state (10)

    The agent can move in four directions (up, down, left, right) and receives rewards based
    on the type of cell it lands on. The environment keeps track of visited states and can
    visualize the agent's path through the grid.

    Attributes:
        room_array (numpy.ndarray): The 2D grid representation of the environment
        action_space (numpy.ndarray): Available actions [0: Up, 1: Down, 2: Left, 3: Right]
        num_states (int): Number of valid states (non-wall cells)
        initial_state (tuple): Starting position coordinates (x, y)
        current_state (tuple): Current position coordinates (x, y)
        visited_states (list): History of visited state coordinates

    Example:
        >>> room_design = [
        ...     [9, 9, 9, 9],
        ...     [9, 10, 0, 9],
        ...     [9, 0, 1, 9],
        ...     [9, 9, 9, 9]
        ... ]
        >>> env = GridWorld(room_design)
        >>> next_state, reward, done = env.step(1)  # Move down
    """

    def __init__(self, room_array: list):
        """Initialize the GridWorld environment.

        Args:
            room_array (list): 2D list representing the grid world where:
                0: Free space
                1: Goal state
                -1: Penalty
                9: Wall
                8: Hallway
                10: Start state

        Raises:
            ValueError: If the room_array doesn't contain a start state (10) or goal state (1)
        """
        self.room_array: NDArray = np.array(room_array)
        self.action_space: NDArray[np.int64] = np.array(
            [0, 1, 2, 3]
        )  # 0:Up, 1:Down, 2:Left, 3:Right
        self.num_actions: int = len(self.action_space)
        self.num_states: int = len(np.where(self.room_array != 9)[0])
        self.initial_state: State = np.where(self.room_array == 10)
        self.current_state: State = self.initial_state
        self.visited_states: List[State] = [self.initial_state]
        self.hallways_info: HallwaysInfo = self.__get_hallways_info()
        self.hallways_indices: List[np.int64] = [
            hallway[-1] for hallway in self.hallways_info
        ]

    def __get_hallways_info(self) -> HallwaysInfo:
        hallways_x, hallways_y = np.where(self.room_array == 8)
        hallways_state = [
            (np.array(x), np.array(y)) for x, y in zip(hallways_x, hallways_y)
        ]
        hallways_info = []
        for hallway_state in hallways_state:
            hallway_state_feature = self.state_to_features(hallway_state)
            hallways_info.append(
                (hallway_state, hallway_state_feature, np.argmax(hallway_state_feature))
            )
        return hallways_info

    def is_hallway_state(self, state: State) -> bool:
        row, col = state[0], state[1]
        return self.room_array[row, col] == 8

    def get_all_states(self) -> list[tuple[int, int]]:
        """Get all valid states (non-wall states) in the grid world.

        Returns:
            list: List of tuples representing the coordinates of valid states
        """
        # Flatten the room_design array and filter out the 9's
        flat_array = [
            (i, j)
            for i, row in enumerate(self.room_array)
            for j, val in enumerate(row)
            if val != 9 and val != 1
        ]
        return flat_array

    def get_random_state(self) -> State:
        """Get a random valid state (non-wall) in the grid world.

        Returns:
            tuple: Random state coordinates as (np.array([x]), np.array([y]))
        """
        all_states = self.get_all_states()
        chosen_state = random.choice(all_states)
        # Convert to numpy arrays to match initial_state format
        return (np.array([chosen_state[0]]), np.array([chosen_state[1]]))

    def state_to_features(self, state: State) -> NDArray:
        """Get the one-hot encoded feature vector for a given state.

        Args:
            state (tuple): State coordinates (x, y)

        Returns:
            numpy.ndarray: One-hot encoded feature vector of length num_states

        Raises:
            ValueError: If the given coordinates don't correspond to a valid state
        """
        # Flatten the room_design array and filter out the 9's
        flat_array = self.get_all_states()

        # Initialize the state_feature array
        state_feature = []

        # Create a zero array with 1 in the corresponding position for each valid cell
        for idx, (i, j) in enumerate(flat_array):
            feature = [0] * len(flat_array)
            feature[idx] = 1
            state_feature.append(feature)

        # Find the index of the given coordinates in the flat_array
        if state in flat_array:
            index = flat_array.index(state)
            return np.array(state_feature[index])
        elif self.is_terminal(state):
            return np.zeros(self.num_states - 1)
        else:
            print("State not found in flat_array:", state)
            raise ValueError(
                "The given coordinates do not correspond to a valid state."
            )

    def features_to_state(self, features: NDArray) -> State:
        """Convert a one-hot encoded feature vector back to state coordinates.

        Args:
            features (numpy.ndarray): One-hot encoded feature vector of length num_states

        Returns:
            tuple: State coordinates as (np.array([x]), np.array([y]))

        Raises:
            ValueError: If the feature vector is invalid or doesn't correspond to a state
        """
        if not any(features) or len(features) != self.num_states - 1:
            raise ValueError("Invalid feature vector")

        # Get the index of the 1 in the one-hot vector
        state_idx = np.argmax(features)

        # Get all valid states in the same order as used in state_to_features
        flat_array = self.get_all_states()

        if state_idx >= len(flat_array):
            raise ValueError("Feature vector index exceeds number of valid states")

        # Get the corresponding coordinates
        x, y = flat_array[state_idx]
        return (np.array([x]), np.array([y]))

    def state_action_to_features(self, state: State, action: np.int64) -> NDArray:
        """Convert a state-action pair to a one-hot encoded feature vector.

        Args:
            state (tuple): Current state coordinates (x, y)
            action (int): Action index (0: Up, 1: Down, 2: Left, 3: Right)

        Returns:
            numpy.ndarray: One-hot encoded feature vector of length (num_states * num_actions)

        Raises:
            ValueError: If the given coordinates don't correspond to a valid state
        """
        # Get all valid states (non-wall states)
        flat_array = self.get_all_states()

        # Find the index of the state in flat_array
        if state in flat_array:
            state_idx = flat_array.index(state)
        else:
            raise ValueError(
                "The given coordinates do not correspond to a valid state."
            )

        # Create the feature vector
        feature_size = (self.num_states - 1) * len(self.action_space)
        features = np.zeros(feature_size)

        # Set the corresponding index to 1
        features[state_idx * len(self.action_space) + action] = 1.0

        return np.array(features)

    def get_reward(self, next_state: State) -> np.int64:
        """Calculate the reward for transitioning to a given state.

        Args:
            next_state (tuple): The state to transition to

        Returns:
            float: The reward value:
                1: Goal state
                0: Start state or wall
                -1: Obstacle
                0: Free space
        """
        try:
            state_reward_value: np.int64 = self.room_array[next_state]
            if (
                state_reward_value == 10
                or state_reward_value == 9
                or state_reward_value == 8
            ):
                return np.int64(0)
            else:
                return state_reward_value
        except IndexError:
            print("Invalid state:", next_state)
            raise IndexError(
                "The given coordinates do not correspond to a valid state."
            )

    def is_terminal(self, state: State) -> bool:
        """Check if the given state is a terminal state (goal).

        Args:
            state (tuple): State coordinates to check

        Returns:
            bool: True if state is the goal state, False otherwise
        """
        return state == np.where(self.room_array == 1)

    def step(self, action: np.int64) -> StepReturns:
        """Execute one time step within the environment.

        Args:
            action (int): The action to take (0: Up, 1: Down, 2: Left, 3: Right)

        Returns:
            tuple: (next_state, reward, done)
                next_state (tuple): The new state coordinates
                reward (float): The reward received
                done (bool): Whether the episode has ended

        Raises:
            ValueError: If the action is not in the action space
        """
        x, y = self.current_state
        if action == 0:
            next_state = (x - 1, y)
        elif action == 1:
            next_state = (x + 1, y)
        elif action == 2:
            next_state = (x, y - 1)
        elif action == 3:
            next_state = (x, y + 1)
        else:
            raise ValueError("Invalid action")
        # Check for wall or obstacle
        if self.room_array[next_state] == 9:
            next_state = (x, y)

        # Add the current state to visited states
        if next_state != self.current_state:  # Only add if actually moving
            self.visited_states.append(next_state)

        self.current_state = next_state
        return next_state, self.get_reward(next_state), self.is_terminal(next_state)

    def reset(self) -> State:
        """Reset the environment to its initial state.

        Returns:
            tuple: The initial state coordinates
        """
        self.current_state = self.initial_state
        self.visited_states = [self.initial_state]
        return self.current_state

    def plot_room(self, file_name=None) -> None:
        """Visualize the grid world and the agent's path.

        Displays a plot showing:
        - The grid world layout with different colors for different cell types
        - The agent's current position (blue triangle)
        - The path taken by the agent (red arrows)

        Args:
            file_name (str, optional): If provided, saves the plot to this file path

        Note:
            - White: Free space
            - Black: Walls
            - Grey: Obstacles
            - Light green: Start state
            - Light coral: Goal state
        """
        # Convert to numpy array and get dimensions
        rows, cols = self.room_array.shape

        # Create RGB array (initialize with black/white)
        rgb_array = np.zeros((rows, cols, 3))  # 3 channels for RGB
        rgb_array[self.room_array == 0] = [1, 1, 1]  # White for 0's
        rgb_array[self.room_array == 9] = [0, 0, 0]  # Black for 1's
        rgb_array[self.room_array == -1] = [0.7, 0.7, 0.7]  # Grey for 2's
        rgb_array[self.room_array == 10] = [
            0.5647,
            0.9333,
            0.5647,
        ]  # Lightgreen for Start
        rgb_array[self.room_array == 1] = [
            0.9412,
            0.5020,
            0.5020,
        ]  # Lightcoral for Start
        rgb_array[self.room_array == 8] = [1.0, 1.0, 0.8]  # Lightyellow for Start

        # # Define colored cells (row, column, color_name)
        # colored_cells = [
        #     (start[0], start[1], 'lightgreen'),     # Central hallway intersection
        #     (goal[0], goal[1], 'lightcoral'),  # Bottom-right corridor
        # ]

        # # Apply colors to specified cells
        # for row, col, color in colored_cells:
        #     rgb_array[row, col] = mcolors.to_rgb(color)

        # Create plot
        plt.figure(figsize=(8, 10))
        plt.imshow(rgb_array, interpolation="nearest", aspect="equal")

        # Configure grid lines
        plt.xticks(np.arange(-0.5, cols, 1))
        plt.yticks(np.arange(-0.5, rows, 1))
        plt.grid(which="major", color="black", linewidth=0.5)
        plt.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )

        # Plot the current state as a triangle
        x, y = self.current_state
        plt.plot(y, x, marker="o", color="blue", markersize=15, label="Current State")

        # Plot arrows for visited states
        for i in range(1, len(self.visited_states)):
            prev_x, prev_y = self.visited_states[i - 1]
            curr_x, curr_y = self.visited_states[i]

            # Determine the direction of movement and plot the corresponding arrow
            if curr_x < prev_x:  # Moved up
                plt.text(
                    prev_y,
                    prev_x,
                    "↑",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=15,
                )
            elif curr_x > prev_x:  # Moved down
                plt.text(
                    prev_y,
                    prev_x,
                    "↓",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=15,
                )
            elif curr_y < prev_y:  # Moved left
                plt.text(
                    prev_y,
                    prev_x,
                    "←",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=15,
                )
            elif curr_y > prev_y:  # Moved right
                plt.text(
                    prev_y,
                    prev_x,
                    "→",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=15,
                )

        # # Add legend
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        if file_name:
            plt.savefig(file_name, transparent=True, bbox_inches="tight")
        plt.show()
