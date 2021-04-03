#python file to generate a complete excel file containing multiple tables
import fixed_point_algo as fixalgo
import numpy as np
import scipy.stats as stats
import initial_conditions as ic
import math

def main():
    '''Power Reward'''
    #implement the algo and print results
    #Test3: Equal vs. Lump-Sum reward
    # B_vals = [0.5]
    # beta_vals = 0.5
    # i_vals = [0.6]
    # # print("Lump Sum Test Started!\n")
    # # fixalgo.full_test_print(B_vals, beta_vals, i_vals, [1, 0, 0], "Outputs/Test3(1):LumpSumReward.csv", alpha_list=ic.alpha_list, reward_type="lump_sum")
    # print("Lump Sum Test Started!\n")
    # fixalgo.full_test_print(B_vals, beta_vals, i_vals, [0, 1, 0], "Outputs/SingleTest.csv",
    #                         alpha_list=ic.alpha_list, reward_group_type="lump_sum")
    '''Piecewise-Constant Reward(for plotting only)'''
    #define initial conditions
    n_list = [i for i in range(5, 21)]
    B = ic.budget
    num_segment = 10
    import os
    for n in n_list:
        n_path = "Outputs/Plots/n = {}".format(n)
        os.mkdir(n_path)
        d_upper = 2 * B / (n-1)
        d_list = [k/num_segment * d_upper for k in range(1, num_segment+1)]
        for d in d_list:
            reward_func = lambda p: ic.piecewise_constant_reward(p, B=B, n=n, d=d)
            reward_groups = [ic.equal_reward_group(reward_func), ic.lump_sum_reward_group(reward_func)]
            fixalgo.single_test_print_plot2(reward_groups, (B, n, d), plot_address=n_path + "/n = {}, d = {}.pdf".format(n, d))
if __name__ == "__main__":
    main()
