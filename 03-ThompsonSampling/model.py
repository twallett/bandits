#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ThompsonSampling:
    """
    Implementation of the Thompson Sampling algorithm for multi-armed bandits.

    Parameters:
    - n_arms (int): The number of arms in the bandit.
    - num_of_params (int): Number of parameters to track for each arm (default is 4).
    - manual_prior (tuple): Tuple containing the initial values for alphas and betas.
      If None, default values of 1 are used for alphas and betas.
    - random_seed (int): Seed for the random number generator (default is None).

    Methods:
    - bandit(data, A, t, n): Simulates pulling an arm and returns the corresponding reward and related information.
    - action(): Chooses an action based on the Thompson Sampling strategy.
    - update(data, A, R, t): Updates the posterior distribution parameters based on the received reward.
    - train(data): Trains the Thompson Sampling algorithm on a given dataset.
    - create_table(): Generates a table with information about arms, action counts, and estimated rewards.
    - save(): Saves the current state of posterior distribution parameters to a DataFrame.
    - recommend(): Recommends an action based on the current estimates (for exploration).
    - reset(): Resets the posterior distribution parameters and other attributes to zeros.

    Attributes:
    - n_arms (int): The number of arms in the bandit.
    - num_of_params (int): Number of parameters to track for each arm.
    - alphas (numpy.ndarray): Parameters for the Beta distribution (positive rewards).
    - betas (numpy.ndarray): Parameters for the Beta distribution (negative rewards).
    - N (numpy.ndarray): Action counts for each arm.
    - Exp (numpy.ndarray): Estimated probabilities of success for each arm.
    - rewards_list (list): List to store received rewards during training.
    - rewards_matrix (numpy.ndarray): Matrix to store rewards for each arm at each time step.
    - param_list (list): List to store posterior distribution parameters at specific time steps.
    """

    def __init__(self, n_arms, num_of_params = 4, manual_prior = None, random_seed = None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.num_of_params = num_of_params
        if manual_prior:
            self.alphas = np.ones((n_arms,1)) * manual_prior[0]
            self.betas = np.ones((n_arms,1)) * manual_prior[1]
        else:
            self.alphas = np.ones((n_arms,1))
            self.betas = np.ones((n_arms,1))
        self.N = np.zeros((n_arms, 1))
        self.Exp = np.zeros((n_arms, 1))
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1,self.n_arms))
        self.param_list = []
        
    def bandit(self, data, A, t, n):
        """
        The function bandit() is assumed to take an action and return a corresponding reward.
        """
        rewards_pull = data.copy()[t:t+1][0]
        R = rewards_pull[A]
        rewards_pull[:A] = 0
        rewards_pull[A+1:] = 0
        
        if type(data[0][0]) == int():
            RG = None
            rewards_pullG = None
        else:
            rewards_pullG = n.inverse_transform(data.copy())
            rewards_pullG = rewards_pullG[t:t+1][0]
            RG = rewards_pullG[A]
            rewards_pullG[:A] = 0
            rewards_pullG[A+1:] = 0
            
        return R, rewards_pull, RG, rewards_pullG

    def action(self):
        Q = np.random.beta(self.alphas, self.betas) 
        return np.argmax(Q)

    def update(self, A, R):
        self.alphas[A] += R
        self.betas[A] += (1 - R)
        
        self.N[A] += 1
        self.Exp = self.alphas / (self.alphas + self.betas)
           
    def train(self, data):
        
        if type(data[0][0]) == int():
            n = None
        else:
            n = MinMaxScaler()
            data = n.fit(data).transform(data)

        for t in range(len(data)):
            
            A = self.action()

            R, rewards_pull, RG, rewards_pullG = self.bandit(data, A, t, n)
            
            if type(data[0][0]) == int():
                self.rewards_matrix = np.vstack([self.rewards_matrix, rewards_pull.reshape(1,-1)])
                self.rewards_list.append(R)
            else:
                self.rewards_matrix = np.vstack([self.rewards_matrix, rewards_pullG.reshape(1,-1)])
                self.rewards_list.append(RG)
                
            self.update(A, R)
        
            if (t+1) % int(len(data) / self.num_of_params) == 0:
                self.param_list.append([t+1, np.hstack((self.alphas, self.betas))])
            
        table = self.create_table()
        
        return table, self.rewards_list, self.rewards_matrix[1:,:], self.param_list

    def create_table(self):
        table = np.hstack([np.arange(1, self.n_arms + 1).reshape(self.n_arms, 1),
                           self.N, self.Exp.round(2)]).astype(float)
        table = pd.DataFrame(data=table, columns=["Arms", "Arm Selection", "E(reward|action)"])
        table = table.to_string(index=False)
        return table

    def save(self):
        df = pd.DataFrame({"Alphas": self.alphas.flatten(), 'Betas': self.betas.flatten()})
        return df

    def reset(self):
        self.alphas = np.ones((self.n_arms,1))
        self.betas = np.ones((self.n_arms,1))
        self.N = np.zeros((self.n_arms, 1))
        self.Exp = np.zeros((self.n_arms, 1))
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1,self.n_arms))
        self.param_list = []
