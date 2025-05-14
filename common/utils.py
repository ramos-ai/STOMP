import os
import pickle
import sys
from datetime import datetime
from typing import Any

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def is_notebook():
    return True if "ipykernel" in sys.modules else False


def get_progress_bar(num_iterations):
    return (
        tqdm_notebook(range(num_iterations))
        if is_notebook()
        else tqdm(range(num_iterations))
    )


def save_learning_results(
    data: Any, experiment_name: str, filename: str, base_path: str = "results"
) -> None:
    """
    Save learning results to a pickle file in a specified folder structure.

    Args:
        data: The data to save (can be list, tuple, or any serializable object)
        experiment_name: Name of the experiment/method (e.g., 'option_learning')
        base_path: Base directory for saving results
    """
    # Create timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the full path including experiment name
    full_path = os.path.join(base_path, experiment_name, timestamp)

    # Create directories if they don't exist
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, filename)

    # Save the data
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
