import numpy as np
import scipy.stats as stats

#import initial conditions from ne
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
width = 10
height = 10

#initial conditions
max_iter = 50
num_stages = 2
num_states = num_stages+1
stage_range = range(1, num_stages + 1) #from 1 to x0
state_range = range(0, num_stages + 1) #from 0 to x0
tol = 0.1
budget = 1
alpha = 0.5

#definte stopping time
T = 3
t_vec = np.linspace(0, T, 100*T)
#p-vector
p_vec = np.linspace(0, 1, 1000)
#test case lists
alpha_list = [alpha/100 for alpha in range(1, 101)]
i_list = [0, 0.2, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
beta_list = [0.1 * i for i in range(2, 11)]
B_list = [0.1 * i for i in range(1, 21)]
#name lists
var_name_list = ["B", "beta", "i"]
#self-chosen cost function
cost = lambda x: 1/2*x**2
#unique maximizer of hamiltonian, lambda^*
maximizer = lambda x: x
#convex conjugate, corresponds to cost function
convex_conjugate = lambda x: maximizer(x)*x - cost(maximizer(x))

'''
Reward Functions
'''
# assume equal difference in consecutive heights, O(n^2)
def piecewise_constant_reward(prop, B=budget, n=10, d=0.1):
    """

    Parameters
    ----------
    B: reward budget
    n : number of partitions
    d : difference in height
    prop : 0 < prop < 1

    Returns
    -------

    """
    if d > 2 * B /(n-1): raise Exception("difference in height too big!")
    initial_height = B + (n - 1)*d/2
    for j in range(n):
        if prop >= j / n and prop < (j + 1) / n:
            return initial_height - j * d
    return 0

#the 'Y' vector for plotting, O(n)
# def piecewise_constant_vector(n, d, prop):

def power_reward(B, beta, i, p):
    return B / (beta ** (i + 1)) * (i + 1) * (beta - p) ** i if p <= beta else 0

#reward_func is pre-filled
def equal_reward_group(reward_func, num_stages=num_stages):
    return [reward_func for stage in range(num_stages)]

#assumes lump-sum reward
def lump_sum_reward_group(reward_func, num_stages=num_stages):
    lump_sum_reward = lambda prop: reward_func(prop) * num_stages
    return [lump_sum_reward if stage == num_stages - 1 else lambda p, stage=stage: 0 for stage in range(num_stages)]

#F_mu_y where 1 <= y <= x0, #cumulative distribution of \mu: R^+ -> [0,1]
F_group_old = F_group = [lambda t, stage=stage: stats.expon.cdf(t, loc=0, scale=stage) for stage in stage_range]
#number of iterations

