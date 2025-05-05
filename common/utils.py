import sys

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
