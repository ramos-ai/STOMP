# SubTask, Option, Model, Planning - STOMP

## Installation

```bash
$ pip install -e .
```

## Introduction

This is an implementation of the STOMP framework, described by (SUTTON et al., 2023) ^[1], which is used in the model-based reinforcement learning to learn options, create environment models based on the learned options and then execute planning with the learned options models.

An important disclaimer is that this this implementation only tried to reproduce the results for the two-rooms environment and didn't reach the exact results described on the paper.

It is important to notice that this is still a work in progress and contributions are welcome.

## Results

The full results can be seen in the main.ipynb notebook and for run the code using a multithread approach, please refer to main.py. Here are a summary of the achieved results

### STOMP Step 2: Option Learning

For the Option Learning plots we achieved a close results as compared to the one on the paper, and the intra-policy that we learn, it is very similar to the one on the paper too.

However, here are two main difference that we can spot in the two images are:

1. In the sutton paper, the start state estimative seems to reach a minimum value of -1.3, while in our implementation our minimum value is -0.2;
2. In the sutton paper, the start state estimative decline more aggressively and the have a "v" shape, while our curve have a less aggressive decline and have something more like a "s" shape.

Nevertheless those differences, we believe that our implementation of Option Learning is correct and reflects the paper description, while the differences pointed here can be attribute to  image scale errors.

| Our Implementation | Original Paper Results |
|:-----------------:|:---------------------:|
| ![Option Learning Results](static/option_learning.png) | ![Sutton's Results](static/option_learning_sutton.png) |

### STOMP Step 3: Model Learning

We did implemented the model learning but we didn't plot the RMS error of such models for lack of clarity on how those errors were calculated.

### STOMP Step 4: Planning

For the planning with options case our plot is not starting at zero, but it is reaching the optimal value and have the same shape of the one presented by the authors. It is not 100 percent clear of what is been plotted on the paper and although the standard deviation of our plot seems huge, it is only a difference of ~0.015.

Therefore, we consider our implementation of the paper is correct.

| Our Implementation | Original Paper Results |
|:-----------------:|:---------------------:|
| ![Planning With Options Results](static/planning_with_options.png) | ![Sutton's Results](static/planning_with_options_sutton.png) |

## Reference

[1] Sutton, R. S., Machado, M. C., Holland, G. Z., Szepesvari, D., Timbers, F., Tanner, B., & White, A. (2023). Reward-respecting subtasks for model-based reinforcement learning. Artificial Intelligence, 324. https://doi.org/10.1016/j.artint.2023.104001
