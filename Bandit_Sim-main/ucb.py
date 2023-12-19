from bandit_sim import Bandit_Sim
from math import sqrt, log
from matplotlib import pyplot as plt
import numpy as np
import random


def ucb(bandit: Bandit_Sim, h: int):
    """Upper Confidence Bound (UCB) algorithm.

    Args:
        bandit (Bandit_Sim): Multi-armed bandit object.
        h (int): Number of bandit arm pulls.

    Returns:
        losses (list): List of losses at each multiple of bandit.n_arms.
    """
    V = 0
    counts = np.zeros(bandit.n_arms)
    means = np.zeros_like(counts)
    losses = []
    for n in range(h):
        if n % bandit.n_arms == 0:
            losses.append(max(bandit.arm_means)*n - V)
        if n < bandit.n_arms:
            arm = n # Pull each arm once at first
        else:
            bounds = [sqrt(2*log(n)/counts[a]) for a in range(bandit.n_arms)]
            arm = np.argmax(means + bounds) # Greedy selection with bounds
        r = bandit.pull_arm(arm)
        V += r
        means[arm] = (counts[arm]*means[arm] + r)/(counts[arm] + 1)
        counts[arm] += 1
    losses.append(max(bandit.arm_means)*h - V)
    return losses

def epsilon_greedy(bandit: Bandit_Sim, h: int, eps: float = .1):
    """Epsilon-greedy algorithm.

    Args:
        bandit (Bandit_Sim): Multi-armed bandit object.
        h (int): Number of bandit arm pulls.
        eps (float, optional): Probability of pulling a random arm. Defaults to .1.

    Returns:
        losses (list): List of losses at each multiple of bandit.n_arms.
    """
    V = 0
    counts = np.zeros(bandit.n_arms)
    means = np.zeros_like(counts)
    losses = []
    for n in range(h):
        if n % bandit.n_arms == 0:
            losses.append(max(bandit.arm_means)*n - V)
        if random.random() < eps:
            arm = random.randrange(bandit.n_arms) # Random selection
        else:
            arm = np.argmax(means) # Greedy selection
        r = bandit.pull_arm(arm)
        V += r
        means[arm] = (counts[arm]*means[arm] + r)/(counts[arm] + 1)
        counts[arm] += 1
    losses.append(max(bandit.arm_means)*h - V)
    return losses


if __name__ == "__main__":
    bandsim = Bandit_Sim(n_arms=2, payout_std=2)
    bandsim.plot(num_samples=10000)

    m = 2000
    h = m*bandsim.n_arms
    losses = []

    losses.append(ucb(bandsim, h))
    losses.append(epsilon_greedy(bandsim, h, .1))
    losses.append(epsilon_greedy(bandsim, h, .2))
    losses.append(epsilon_greedy(bandsim, h, .3))

    plt.plot(list(zip(*losses)))
    plt.xlabel("Number of Episodes (m)")
    plt.ylabel("Cumulative Regret")
    plt.legend(["UCB", "eps=.1", "eps=.2", "eps=.3"])
    plt.show()
