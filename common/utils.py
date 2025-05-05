import sys


def is_notebook():
    return True if "ipykernel" in sys.modules else False
