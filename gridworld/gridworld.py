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

        # Find the index of the given coordinates in the flat_array
        if state in flat_array:
            state_features = np.zeros(len(flat_array))
            index = flat_array.index(state)
            state_features[index] = 1.0
            return np.array(state_features)  # (state_feature[index])
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


# keeping the original class for now

from enum import IntEnum
from os.path import join
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

State = Tuple[int, int]  # (x, y) coordinates in the grid
OneHotVector = NDArray[np.int_]


class StateType(IntEnum):
    FREE = 0
    WALL = 9
    CURRENT = -10
    START = 10
    GOAL = 100
    PENALTY = -1
    BOTTLENECK = 8


class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class TemporaryGridWorld:
    def __init__(
        self,
        room_array: List[List[int]],
        initial_and_goal_states: Tuple[State, State] | None = None,
        success_prob: float = 1.0,
    ):
        self.room_array: NDArray[np.int_] = np.array(room_array)
        self.reward_map: NDArray[np.int_] = np.array(room_array)
        self.room_height, self.room_width = self.room_array.shape
        if not 0 <= success_prob <= 1:
            raise ValueError("success_prob must be between 0 and 1")
        self.success_prob = success_prob

        # Setting States
        self.initial_state, self.goal_state = self.__get_initial_and_goal_states(
            initial_and_goal_states
        )

        # Setting maps
        self.state_idx_to_coordinates, self.state_coordinates_to_idx = (
            self.get_state_maps()
        )

        # Basic properties
        self.num_states = len(self.state_idx_to_coordinates)
        self.num_actions = len(Actions)

        self.current_state: State = self.initial_state
        self.done = False

    def save_room(self, base_path: str):
        np.save(join(f"{base_path}", "room_array.npy"), self.room_array)
        np.save(join(f"{base_path}", "reward_map.npy"), self.reward_map)

    def get_state_maps(
        self,
        states_to_exclude: List[StateType] = [StateType.WALL, StateType.GOAL],
        set_as_property: bool = True,
    ) -> Tuple[Dict[int, State], Dict[State, int]]:
        state_idx_to_coordinates: Dict[int, State] = {}
        state_coordinates_to_idx: Dict[State, int] = {}
        count = 0
        for i in range(self.room_height):
            for j in range(self.room_width):
                if self.room_array[i][j] not in states_to_exclude:
                    state_idx_to_coordinates[count] = (j, i)
                    state_coordinates_to_idx[(j, i)] = count
                    count += 1
        if set_as_property:
            self.state_idx_to_coordinates = state_idx_to_coordinates
            self.state_coordinates_to_idx = state_coordinates_to_idx
            self.num_states = len(self.state_idx_to_coordinates)
        return state_idx_to_coordinates, state_coordinates_to_idx

    def reset(
        self,
        states_to_exclude: List[StateType] = [StateType.WALL, StateType.GOAL],
        set_as_property: bool = True,
    ) -> State:
        self.get_state_maps(states_to_exclude, set_as_property)
        self.done = False
        self.current_state = self.initial_state
        return self.initial_state

    def is_terminal(self) -> bool:
        if self.current_state == self.goal_state:
            self.done = True
        return self.done

    def get_reward(self, next_state) -> float:
        if next_state == self.goal_state:
            return 1.0
        elif self.reward_map[next_state[1]][next_state[0]] == StateType.PENALTY:
            return -1.0
        else:
            return 0.0

    def step(self, action: Actions) -> Tuple[State, float, bool]:
        if np.random.random() > self.success_prob:
            available_actions = [a for a in Actions if a != action]
            action = np.random.choice(available_actions)

        x, y = self.current_state

        if action == Actions.UP:
            next_state = (x, y - 1)
        elif action == Actions.DOWN:
            next_state = (x, y + 1)
        elif action == Actions.LEFT:
            next_state = (x - 1, y)
        elif action == Actions.RIGHT:
            next_state = (x + 1, y)
        else:
            raise ValueError(f"Invalid action: {action}")

        if self.room_array[next_state[1]][next_state[0]] == StateType.WALL:
            next_state = self.current_state

        self.current_state = next_state

        return next_state, self.get_reward(next_state), self.is_terminal()

    def get_one_hot_state(self, state: State) -> OneHotVector:
        one_hot_state = np.zeros(self.num_states, dtype=np.int_)
        idx = self.state_coordinates_to_idx[state]
        one_hot_state[idx] = 1
        return one_hot_state

    def get_state_from_one_hot(self, one_hot_state: OneHotVector) -> State:
        idx = int(np.argmax(one_hot_state))
        return self.state_idx_to_coordinates[idx]

    def get_one_hot_state_action(self, state: State, action: Actions) -> OneHotVector:
        one_hot_state_action = np.zeros(
            self.num_states * self.num_actions, dtype=np.int_
        )
        state_idx = self.state_coordinates_to_idx[state]
        action_idx = int(action)
        idx = state_idx * self.num_actions + action_idx
        one_hot_state_action[idx] = 1
        return one_hot_state_action

    def get_state_action_from_one_hot(
        self, one_hot_state_action: OneHotVector
    ) -> Tuple[State, Actions]:
        idx = int(np.argmax(one_hot_state_action))
        state_idx = idx // self.num_actions
        action_idx = idx % self.num_actions
        state = self.state_idx_to_coordinates[state_idx]
        action = Actions(action_idx)
        return state, action

    def plot_room(self, file_name=None) -> None:
        # Create RGB array (initialize with black/white)
        rgb_array = self.__get_rgb_array()

        # Create plot
        plt.figure(figsize=(24, 30))
        plt.imshow(rgb_array, interpolation="nearest", aspect="equal")

        # Configure grid lines
        plt.xticks(np.arange(-0.5, self.room_width, 1))
        plt.yticks(np.arange(-0.5, self.room_height, 1))
        plt.grid(which="major", color="black", linewidth=0.5)
        plt.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )

        x, y = self.current_state
        plt.plot(x, y, marker="o", color="blue", markersize=15, label="Current State")

        if file_name:
            plt.savefig(file_name, transparent=True, bbox_inches="tight")
        plt.show()

    def plot_policy(
        self, policy_probs: NDArray[np.floating], file_name=None, only_max: bool = False
    ) -> None:
        # Reshape flattened policy into (num_states, num_actions)
        policy_matrix = policy_probs.reshape(self.num_states, self.num_actions)
        rgb_array = self.__get_rgb_array()

        plt.figure(figsize=(24, 30))
        plt.imshow(rgb_array, interpolation="nearest", aspect="equal")

        # Arrow properties
        arrow_props = dict(
            head_width=0.3, head_length=0.3, fc="blue", ec="blue", alpha=0.7
        )

        # Dictionary for action directions
        action_vectors = {
            Actions.UP: (0, -1),
            Actions.DOWN: (0, 1),
            Actions.LEFT: (-1, 0),
            Actions.RIGHT: (1, 0),
        }

        for state_idx, coords in self.state_idx_to_coordinates.items():
            x, y = coords
            state_probs = policy_matrix[state_idx]
            # Softmax to get probabilities
            exp_probs = np.exp(state_probs)
            probs = exp_probs / exp_probs.sum()

            if only_max:
                # Plot only the action with highest probability
                max_action = np.argmax(probs)
                prob = probs[max_action]
                dx, dy = action_vectors[Actions(max_action)]
                arrow_length = prob * 0.4
                plt.arrow(x, y, dx * arrow_length, dy * arrow_length, **arrow_props)
            else:
                # Plot arrows for actions with probability > 0.1
                for action in Actions:
                    prob = probs[action]
                    if prob > 0.1:  # Only show significant probabilities
                        dx, dy = action_vectors[action]
                        # Scale arrow length by probability
                        arrow_length = prob * 0.4
                        plt.arrow(
                            x, y, dx * arrow_length, dy * arrow_length, **arrow_props
                        )

        # Configure grid
        plt.xticks(np.arange(-0.5, self.room_width, 1))
        plt.yticks(np.arange(-0.5, self.room_height, 1))
        plt.grid(True, color="black", linewidth=0.5)
        plt.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )

        if file_name:
            plt.savefig(file_name, transparent=True, bbox_inches="tight")
        plt.show()

    def plot_room_with_states(self, file_name=None) -> None:
        # Create RGB array (initialize with black/white)
        rgb_array = self.__get_rgb_array()

        # Create plot
        plt.figure(figsize=(24, 30))
        plt.imshow(rgb_array, interpolation="nearest", aspect="equal")

        # Configure grid lines
        plt.xticks(np.arange(-0.5, self.room_width, 1))
        plt.yticks(np.arange(-0.5, self.room_height, 1))
        plt.grid(which="major", color="black", linewidth=0.5)
        plt.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )

        # Add state identifications
        for state_idx, coords in self.state_idx_to_coordinates.items():
            x, y = coords
            # Skip walls and goal states as they are not in the state mapping
            if self.room_array[y][x] not in [StateType.WALL, StateType.GOAL]:
                plt.text(
                    x,
                    y,
                    f"{state_idx}: {coords}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    rotation=45
                )

        # Mark the goal state
        gx, gy = self.goal_state
        plt.text(
            gx,
            gy,
            "GOAL",
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            weight='bold'
        )

        # Mark current state with a blue dot
        x, y = self.current_state
        plt.plot(x, y, marker="o", color="blue", markersize=15, label="Current State")

        if file_name:
            plt.savefig(file_name, transparent=True, bbox_inches="tight", dpi=300)
        plt.show()

    def __get_rgb_array(self) -> NDArray[np.floating]:
        rgb_array = np.zeros((self.room_height, self.room_width, 3))
        rgb_array[self.room_array == StateType.FREE] = [1, 1, 1]  # White
        rgb_array[self.room_array == StateType.WALL] = [0, 0, 0]  # Black
        rgb_array[self.room_array == StateType.PENALTY] = [0.7, 0.7, 0.7]  # Grey
        rgb_array[self.room_array == StateType.BOTTLENECK] = [
            1.0,
            1.0,
            0.8,
        ]  # Light Yellow
        rgb_array[self.room_array == StateType.START] = [
            0.5647,
            0.9333,
            0.5647,
        ]  # Lightgreen
        rgb_array[self.room_array == StateType.GOAL] = [
            0.9412,
            0.5020,
            0.5020,
        ]  # Lightcoral
        return rgb_array

    def __get_initial_and_goal_states(
        self, initial_and_goal_states: Tuple[State, State] | None = None
    ) -> Tuple[State, State]:
        if initial_and_goal_states is not None:
            initial_state, goal_state = initial_and_goal_states
            self.room_array[initial_state[1]][initial_state[0]] = StateType.START
            self.room_array[goal_state[1]][goal_state[0]] = StateType.GOAL
            return initial_state, goal_state
        while True:
            initial_state = self.__get_specific_state(StateType.START)
            goal_state = self.__get_specific_state(StateType.GOAL)
            if initial_state != goal_state:
                break
        return initial_state, goal_state

    def __get_specific_state(self, state_type: StateType) -> State:
        y, x = np.where(self.room_array == state_type)
        if y.size == 0 or x.size == 0:
            random_state = self.__get_random_state()
            self.room_array[random_state[1]][random_state[0]] = state_type
            return random_state
        elif y.size == 1 or x.size == 1:
            return (int(x[0]), int(y[0]))
        else:
            raise ValueError(
                f"Multiple states {state_type.name} found in the room array."
            )

    def __get_random_state(self) -> State:
        while True:
            x = np.random.randint(self.room_height - 1) + 1
            y = np.random.randint(self.room_width - 1) + 1
            if self.room_array[x][y] == 0:
                return (y, x)
