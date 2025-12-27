# ruff: noqa
# pyright: basic, reportUnusedImport=false, reportUnusedParameter=false, reportUnusedVariable=false

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np


def run_bandit_experiment(
    k: int = 10,
    steps=200000,
    runs=2000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    drift_std=0.01,
    baseline: int = 0,
):
    """Run the bandit experiment

    Args:
        k (int): The number of arms for the testbed
        steps (int): The number of consecutive choices made by the agent
    """
    true_q = np.zeros((runs, k))
    optimal_actions = np.zeros((runs, steps))
    rewards_sample_avg = np.zeros((runs, steps))  
    rewards_const_step = np.zeros((runs, steps))  

    # sample average method
    Q_sample = np.full((runs, k), baseline, float)
    N_sample = np.zeros((runs, k), int)

    # const alp stepsize method
    Q_const = np.full((runs, k), baseline)

    for t in range(steps):
    # random walk to update the non stationary 
        true_q +=  np.random.normal(0, drift_std, size=(runs, k))
        optimal_actions = np.argmax(true_q, axis=1)

        # whether to explore each cell 
        explore = np.random.rand(runs) < epsilon
        random_arm = np.random.randint(0, k, runs)

        greedy_actions = np.argmax(Q_sample, axis=1)
        actions_sample_avg = np.where(explore, random_arm, greedy_actions)

        # step size updates
        greedy_actions = np.argmax(Q_const, axis=1)

def main():
    print("Hello from bandit-methods!")


if __name__ == "__main__":
    main()
