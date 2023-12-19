import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys

class Bandit_Sim:
    """A class used for simulating multi-armed bandit problems

    Description
    -----------
    Uses a list of normal distributions with mean (i+1)/(`n_arms`+1) where i is in [0, `n_arms`). 
    Each arm distribution has a standard deviation of `payout_std`.

    Attributes
    ----------
    `n_arms`      : int
        the number of arms to be simulated
    `payout_std`  : float
        the standard deviation for the payout of each arm
    `arm_means`   : list<floats>
        a list containing the mean payouts for each arm
    """
    def __init__(self, n_arms, payout_std, seed=None):
        """Function to initialize a bandit simulator
        
       Parameters
       ----------
        `n_arms`      : int
            the number of arms to be simulated
        `payout_std`  : float
            the standard deviation for the payout of each arm.
        `seed`        : int, optional
            the random seed
        """
        np.random.seed(seed=seed)
        self.n_arms = n_arms
        self.payout_std = payout_std
        self.arm_means = [(i+1)/(self.n_arms+1) for i in range(self.n_arms)]

    def pull_arm(self, n):
        """A method which returns a payout from arm `n`

        Parameters
        ----------
        `n`           : int
            the index of the arm to pull, must be in interval [0, `n_arms`)

        Returns
        -------
        payout      : float
            the reward for pulling arm n
        """
        return np.random.normal(self.arm_means[n], self.payout_std)
    
    def plot(self, num_samples):
        """A method to visualize the distributions of the arms and the joint distribution of payouts

        Description
        -----------
        This method will display two plots, one showing the joint histogram of the combined payouts
        from all arm pulls, and the other showing the payout for each arm separately, with each arm 
        shown in a different color.

        Parameters
        ----------
        `num_samples` : int
            the number of samples to draw for each arm
        """
        fill_colors, edge_colors = self._gen_color_list()
        arms_data = []
        for i in range(self.n_arms):
            arms_data.append([[self.pull_arm(i) for _ in range(num_samples)]])
        full_data = np.array(arms_data).flatten()
        _, bins, _ = plt.hist(full_data, bins=int((self.n_arms*num_samples)**(1/2.5)))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(self.n_arms):
            ax.hist(arms_data[i], histtype='stepfilled', fc=fill_colors[i], bins=bins)
            ax.hist(arms_data[i], histtype='step', edgecolor=edge_colors[i], bins=bins)
        
        plt.show()
    
    def _gen_color_list(self):
        """Generates the colors for each arm plot"""
        num_steps = self.n_arms
        hue = 0.0
        step_val = 1/num_steps + 0.25
        fill_colors = []
        edge_colors = []
        for _ in range(num_steps):
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            hue += step_val
            hue %= 1.0 # cap hue at 1.0
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            fill_colors.append((r, g, b, .33))
            edge_colors.append((r, g, b, 1))
        
        return fill_colors, edge_colors        
