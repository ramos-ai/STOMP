import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gridworld.gridworld import GridWorld
from gridworld.room_design import two_room_design as room_design
from stomp.framework import STOMP


num_runs = 100
all_runs = []


def run_experiment(run_idx):
    env = GridWorld(room_design)
    stomp = STOMP(
        env,
        experiment_name=f"reward_respecting_options_linear_combination_{run_idx}",
    )

    env.reset()
    option_learning_logs, options_model_learning_logs, planning_logs = stomp.learn(
        log_freq=None
    )
    return (option_learning_logs, options_model_learning_logs, planning_logs)


num_processes = os.cpu_count() - 1
print(f"Running with {num_processes} processes\n")

with Pool(processes=num_processes) as pool:
    all_runs = list(tqdm(pool.map(run_experiment, range(num_runs)), total=num_runs))

learning_options_start_state_estimative = []
planning_start_state_estimative = []
for run in all_runs:
    # Let's consider only the two rooms experiment
    # Therefore we have only one hallway
    learning_options_start_state_estimative.append(run[0][0])
    planning_start_state_estimative.append(run[2])
learning_options_start_state_estimative = np.stack(
    learning_options_start_state_estimative
)
planning_start_state_estimative = np.stack(planning_start_state_estimative)
learning_options_start_state_estimative.shape, planning_start_state_estimative.shape

learning_options_start_state_estimative_mean = np.mean(
    learning_options_start_state_estimative, axis=0
)
learning_options_start_state_estimative_std = np.std(
    learning_options_start_state_estimative, axis=0
)
(
    learning_options_start_state_estimative_mean.shape,
    learning_options_start_state_estimative_std.shape,
)

planning_start_state_estimative_mean = np.mean(planning_start_state_estimative, axis=0)
planning_start_state_estimative_std = np.std(planning_start_state_estimative, axis=0)
planning_start_state_estimative_mean.shape, planning_start_state_estimative_std.shape


def plot_arrays(mean_array, std_array, plotting_info, plotting_name):
    # Create figure and axis
    plt.figure(figsize=(20, 6))

    # Generate x-axis points (assuming these are sequential steps/episodes)
    x = np.arange(len(mean_array))

    # Plot mean line with shaded standard deviation
    plt.plot(x, mean_array, "b-", label="Mean")
    plt.fill_between(
        x, 
        mean_array - std_array,
        mean_array + std_array,
        color="b",
        alpha=0.2,
        label="Standard Deviation",
    )

    # Customize the plot
    plt.xlabel(plotting_info["xlabel"])
    plt.ylabel(f"{plotting_info['ylabel']}\n(Average Over 100 runs)")
    plt.title(plotting_info["title"])
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()
    plt.savefig(f"{plotting_name}.png", bbox_inches="tight")


plot_arrays(
    learning_options_start_state_estimative_mean,
    learning_options_start_state_estimative_std,
    {
        "xlabel": "Steps of Off-Policy Experience",
        "ylabel": "Initial State Estimative: v_hat(s0)",
        "title": "STOMP Step 2: Option Learning",
    },
    "option_learning",
)

plot_arrays(
    planning_start_state_estimative_mean,
    planning_start_state_estimative_std,
    {
        "xlabel": "Number of Planning Look-ahead Operations",
        "ylabel": "Initial State Estimative: v_hat(s0)",
        "title": "STOMP Step 4: Planning with Options",
    },
    "planning_with_options",
)
