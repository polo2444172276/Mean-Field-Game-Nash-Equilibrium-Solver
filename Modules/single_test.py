#python file to test specific outcome cases of the fixed point algo
import fixed_point_algo as fixalgo
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import initial_conditions as ic
import math

def single_power_model_test():

    #do a single test case
    B = 0.5
    beta = 0.5
    i = 0.8
    # equal_reward_group = [lambda prop: fixalgo.equal_reward(B, beta, i, prop) for stage in ic.stage_range]
    # fixalgo.single_test_print(equal_reward_group, B=B, beta=beta, i=i, print_option=False)
    # print("\n")

    lump_sum_reward_group = [lambda prop, stage=stage: fixalgo.lump_sum_reward(B, beta, i, prop, stage) for stage in ic.stage_range]
    # fixalgo.single_test_print(lump_sum_reward_group, B=B, beta=beta, i=i, print_option=True, plot_option=False,output_address="Outputs/SingleTest.csv")

    '''plotting'''

    F_group_matrix, num_iters = fixalgo.calc_F_eq(lump_sum_reward_group)
    F_group_eq = F_group_matrix[-1]
    # update order: F_old -> g -> v -> lambda -> ro -> F_new
    g_group_eq = fixalgo.solve_g_group(F_group_eq, lump_sum_reward_group)
    v_group_eq = fixalgo.solve_v_group(g_group_eq)
    lambda_group_eq = fixalgo.solve_lambda_group(g_group_eq, v_group_eq, ic.maximizer)
    ro_group_eq = fixalgo.solve_ro_group(lambda_group_eq)
    F_group_new = fixalgo.solve_F_group(ro_group_eq)


    import matplotlib.pyplot as plt
    #first graph is reward functions
    plt.subplot(2, 2, 1)
    reward_labels = ["reward_{}".format(i + 1) for i in range(ic.num_stages)]
    fixalgo.plot_group("reward_functions", lump_sum_reward_group, ic.colors, labels=reward_labels, t_vec=np.linspace(0, 1, 100))
    #second graph is initial cdf F's
    plt.subplot(2, 2, 2)
    iter = 1
    F_start_labels = ["F_start".format(i + 1) for i in range(ic.num_stages)]
    fixalgo.plot_group("F's at iteration {}".format(iter), ic.F_group_old, ic.colors, labels=F_start_labels)
    # third graph is F's at second last iteration
    plt.subplot(2, 2, 3)
    effort_labels = ["effort_{}".format(i + 1) for i in range(ic.num_states)]
    fixalgo.plot_group("Optimal effort's", lambda_group_eq, ic.colors, labels=effort_labels)

    # fourth graph is F's at last iteration, i.e. eq
    plt.subplot(2, 2, 4)
    F_final_labels = ["F_final_{}".format(i + 1) for i in range(ic.num_stages)]
    fixalgo.plot_group("Equilibrium Cdf's", F_group_eq, ic.colors, labels=F_final_labels)
    plt.show()

def piecewise_constant_reward_test():
    n = 10
    d1, d2 = 0.01, 0.02
    piecewise_constant1 = [(lambda p, i=i: fixalgo.piecewise_constant(p, n, d1)) if i == ic.num_stages - 1 else lambda p: 0 for
                           i in range(ic.num_stages)]

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    reward_labels = ["reward_{}".format(i + 1) for i in range(ic.num_stages)]
    fixalgo.plot_group("piecewise_constant1, d=0.01", piecewise_constant1, ic.colors, labels=reward_labels,
               t_vec=np.linspace(0, 1, 1000))

    piecewise_constant2 = [(lambda p, i=i: fixalgo.piecewise_constant(p, n, d2)) if i == ic.num_stages - 1 else lambda p: 0 for
                           i in range(ic.num_stages)]
    plt.subplot(1, 2, 2)
    reward_labels = ["reward_{}".format(i + 1) for i in range(ic.num_stages)]
    fixalgo.plot_group("piecewise_constant2, d=0.02", piecewise_constant2, ic.colors, labels=reward_labels,
               t_vec=np.linspace(0, 1, 1000))
    plt.show()

def main():
    import matplotlib.pyplot as plt
    pw_test_vec = [ic.piecewise_constant_reward(p, B=1, n=10, d=0.1) for p in np.linspace(0, 1, 100)]
    reward_func = lambda p: ic.piecewise_constant_reward(p, B=1, n=10, d=0.2)
    eq_reward_group = ic.equal_reward_group(reward_func)
    ls_reward_group = ic.lump_sum_reward_group(reward_func)
    reward_groups = (eq_reward_group, ls_reward_group)
    labels = ["test_{}".format(num) for num in ic.stage_range]
    fixalgo.single_test_print_plot2(reward_groups, plot_option=True, plot_address="Outputs/Plots/test.pdf")
    # fixalgo.plot_group("test pw func", reward_group, ic.colors, labels, t_vec=ic.p_vec)
    plt.show()

if __name__ == "__main__":
    main()
