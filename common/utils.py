import os
import pickle
import sys
import numpy as np
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

    # Create the full path including experiment name
    full_path = os.path.join(base_path, experiment_name)

    # Create directories if they don't exist
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, filename)

    # Save the data
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def save_model(model: np.ndarray, filename: str, base_path: str = "./models/") -> None:
    """
    Save a model (numpy array) to a specified file path.

    Args:
        model: The model to save (numpy array)
        filename: Name of the file to save the model
        base_path: Base directory for saving models
    """
    # Create the full path
    full_path = os.path.join(base_path, filename)

    # Create directories if they don't exist
    os.makedirs(base_path, exist_ok=True)

    # Save the model
    np.save(full_path, model)


def load_model(filename: str, base_path: str = "./models/") -> np.ndarray:
    """
    Load a model (numpy array) from a specified file path.

    Args:
        filename: Name of the file to load the model from
        base_path: Base directory for loading models

    Returns:
        The loaded model as a numpy array
    """
    full_path = os.path.join(base_path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file {full_path} does not exist.")

    return np.load(full_path)


if __name__ == "__main__":
    # Example usage
    model = np.array([[1, 2], [3, 4]])
    save_model(model, "example_model.npy")

    loaded_model = load_model("example_model.npy")
    print("Loaded model:", loaded_model)
