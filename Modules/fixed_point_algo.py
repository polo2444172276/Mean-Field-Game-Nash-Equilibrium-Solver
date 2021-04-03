'''This module contains all the functions associated with the fixed point algorithm'''
import numpy as np
from scipy.integrate import odeint
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import initial_conditions as ic
import copy

#initial conditions
num_stages = ic.num_stages

'''0. Helper/test functions'''
#plot groups of functions in the same graph
def plot_group(title, fun_group, colors, labels, t_vec=ic.t_vec):
    # plt.figure()
    if title != "": plt.title(title, fontsize=5)
    for i in range(len(fun_group)):
        func = fun_group[i]
        fun_vec = [func(t) for t in t_vec]
        plt.plot(t_vec, fun_vec, color=colors[i], label=labels[i])
        plt.legend(loc='best', prop={"size": 5})

    # plt.show()

#calculates the "distance" between two vectors
def vec_distance(y1_vec, y2_vec):
    return sum([(y1_vec[i] - y2_vec[i])**2 for i in range(len(y1_vec))])
#calculates the "distance" between two functions, through discretizing domain and sum of squares
def func_distance(x_vec, y1_func, y2_func):
    # x_vec:partition of [a,b]; y1_func, y2_func are functions of x
    y1_vec = [y1_func(x) for x in x_vec]
    y2_vec = [y2_func(x) for x in x_vec]
    return vec_distance(y1_vec, y2_vec)
#calculates the "distance" between groups of functions, returns a group
def func_group_distance(x_vec, y1_group, y2_group):
    return [func_distance(x_vec, y1_group[i], y2_group[i]) for i in range(len(y1_group))]
#calculates the "variance" of a vector
def vec_var(y_vec):
    n = len(y_vec)
    mean = sum(y_vec)/n
    return sum([(mean - y)**2 for y in y_vec])/n
#calculates the "variance" of a function on a given range
def func_var(x_vec, y_func):
    y_vec = [y_func(x) for x in x_vec]
    return vec_var(y_vec)

#F is an increasing function on domain x_vec
# returns F^(-1)(prop) if it is in x_vec
# return -1 if desired result is not in x_vec
def quantile(F, x_vec, prop):
    for i in range(len(x_vec) - 1):
        curr, next = x_vec[i], x_vec[i+1]
        if (prop >= F(curr) and prop <= F(next) ) :
            return curr
    return +100.
'''Formal Algorithm Region'''
'''1.Define helper functions'''

#defines finite difference method (2)
def trapezoid_integral(dydx, y0, x):
    """
    solves for an ode using trapezoid integration
    Parameters
    ----------
    dydx(y, x): differential equation for y;
    y0 : initial value
    x : partition of [a,b], i.e.list/vec containing equal-spaced x values;

    Returns a list/vec containing solved y-values
    -------

    """
    h = x[1] - x[0]
    y = [y0]
    n = len(x)
    y.append(y[0] + h * dydx(y[0], x[0]))
    dy1, dy2 = dydx(y[0], x[0]), dydx(y[1], x[1])
    # y[1] = y0 + 1/2*h*(dy1 + dy2) #adjust y_1 using trapezoid rule
    for i in range(1, n - 1):
        dy1 = dydx(y[i - 1], x[i - 1])
        dy2 = dydx(y[i], x[i])
        #idea of this is to use yn'+ (yn'-yn-1')/2 to approximate h*(yn+1 - yn)
        #very similar to trapezoid integration
        y.append(y[i] + h * dy2 + 1/2*h*(dy2 - dy1))
    return y
#use odeint as default way of solving odes, adjust the output format to a normal list
def odeint_method(dydx, y0, t_vec):
    return [y[0] for y in odeint(dydx, y0, t_vec)]
#plots y to x in a separate graph with given information
def self_plot(title, x, y, color, label):
    plt.figure(title)
    plt.plot(x, y, color, label=label)
    plt.legend(loc="best")

#use trapezoid FDM with O(h^2) error
integ_method = trapezoid_integral
'''2.Define intermediate functions'''
''' R,F_old -> g -> v -> lambda -> ro -> F_new'''

'''R,F_old -> g'''
#solve a list of g's given a list of F's
def solve_g_group(F_group, reward_group):
    g_group = []
    for stage in ic.stage_range:
        g_func = lambda t, stage=stage: reward_group[stage-1](F_group[stage-1](t))
        g_group.append(g_func)
    return g_group

'''g -> v'''
#g_next = g_y+1(t), v_next = v_y+1(t), t_vec: list of t-values, v(T) = v0
#returns v_y(t)
def solve_v(g_next, v_next, t_vec=ic.t_vec,v0 = 0):
    #define the ode of v(t) using time reverse w(t) = v(T-t)
    ode_v_curr = lambda v, t: ic.convex_conjugate(g_next(ic.T - t) + v_next(ic.T - t) - v)
    v_curr_reversed = integ_method(ode_v_curr, v0, t_vec)
    n = len(t_vec)
    #reverse the vector v(T-t) back to v(t)
    v_curr = [v_curr_reversed[n - 1 - i] for i in range(n)]
    #interpolate vector to make a function
    v_curr_func = interp1d(t_vec, v_curr, kind="cubic")
    return v_curr_func

#solve for list of v given list of g, where g_group[y] = g_y+1(t)
#returns a list of value functions such that v_group[y] = v_y+1(t)
def solve_v_group(g_group):
    v_group = [lambda t:0]# v_xo(t) = 0
    for state in range(ic.num_stages-1, -1, -1): # x0 - 1 >= state >= 0
        v_next = v_group[0]
        g_next = g_group[state]
        v_curr = solve_v(g_next, v_next, v0=0)
        #insert it to front position because v_y is solved in backwards direction
        v_group.insert(0, v_curr)
    return v_group

'''v -> lambda'''
#returns lamstar_y(t)
def lambda_star(g, stage):
    #precondition: t > 0, 1 <= stage <= x0
    if stage >= ic.num_stages: return lambda t: 0
    else:
        g_next = lambda t: g(t, stage+1)
        v_next = ic.solve_for_v(g, stage+1)
        v_curr = ic.solve_for_v(g, stage)
        return lambda t: ic.maximizer(g_next(t) + v_next(t) + v_curr(t))

#solve lambda given list of g, list of v and Big lambda
#g_group[y] = g_y+1(t), v_group(y) = v_y+1(t) for all y
def solve_lambda_group(g_group, v_group, maximizer):
    lambda_group = []
    for state in range(0, ic.num_stages): # 0 <= state <= x0 - 1
        # general equation:lambda_y(t) = maximizer(g_y+1(t) + v_y+1(t) - v_y(t) )
        #indexing in g_group is different from indexing in v_group
        lambda_func = lambda t, state=state: maximizer(g_group[state](t)
                        + v_group[state+1](t) - v_group[state](t))
        lambda_group.append(lambda_func)
    lambda_group.append(lambda t:0) #lambda_x0(t) = 0
    return lambda_group

'''lambda -> ro'''
#solve for list of ro given list of lambda
def solve_ro_group(lambda_group, t_vec=ic.t_vec):
    #ro_0'(t) = -lambda_0(t) * ro_0(t)
    ro_ode1 = lambda ro, t: -lambda_group[0](t)*ro
    ro0_vec = integ_method(ro_ode1, 1, t_vec)
    ro0_func = interp1d(t_vec, ro0_vec, kind="cubic")
    #append ro_1(t) to ro_group
    ro_group = [ro0_func]

    #solve for ro_y(t) for 1 <= y <= x0
    for state in range(1, num_stages + 1):
        #ro_y'(t) = lambda_y-1(t)*ro_y-1(t) - lambda_y(t)*ro_y(t) for y > 1
        ro_ode_y = lambda ro, t, state=state:\
         lambda_group[state-1](t)*ro_group[state-1](t) - lambda_group[state](t)*ro
        ro_y_vec = integ_method(dydx=ro_ode_y, y0=0, x=t_vec)
        ro_y_func = interp1d(t_vec, ro_y_vec, kind="cubic")
        ro_group.append(ro_y_func)
    return ro_group

'''ro -> F'''
#solves for a single F_new function
def solve_F_func(ro_group, stage_num):
    return lambda t: sum([ro_group[stage](t) for stage in range(stage_num, num_stages+1)])

# precondition: ro_group[y] = ro_y(t)
#solve F_y for 1 <= y <= x0
def solve_F_group(ro_group):
    return [solve_F_func(ro_group, stage) for stage in range(1, num_stages+1)]

'''3. gather all functions in 2 into one function'''
#update one iteration
#returns all updated groups in a tuple
def update_F_group(F_group_old, reward_group):
    # update order: F_old -> g -> v -> lambda -> ro -> F_new
    g_group = solve_g_group(F_group_old, reward_group)
    v_group = solve_v_group(g_group)
    lambda_group = solve_lambda_group(g_group, v_group, ic.maximizer)
    ro_group = solve_ro_group(lambda_group)
    F_group_new = solve_F_group(ro_group)
    return F_group_new
#returns a group of equilibrium cdfs of F and the number of iterations
def calc_F_eq(reward_group, max_iter=ic.max_iter,F_initials=ic.F_group_old, tol = ic.tol, t_vec=ic.t_vec):
    # eventually distance_matrix should be of size num_iters * num_stages
    distance_matrix = []
    F_group_matrix = [F_initials]
    num_iter = 0
    F_group_old = copy.deepcopy(F_initials)

    #do at least two iterations, stop when reaching maximum number of iterations or average of distances
    # between functions in the last two turn <= tol
    while not(num_iter >= 2 and
              ((num_iter >= max_iter) or (sum(distance_matrix[-1] + distance_matrix[-2]) < 2 * num_stages * tol))):
        F_group_new = update_F_group(F_group_old, reward_group=reward_group)
        F_group_matrix.append(F_group_new)
        distance_matrix.append(func_group_distance(t_vec, F_group_old, F_group_new))
        print("Distance between F_old and F_new at iteration{} is:{}".
              format(num_iter + 1, distance_matrix[-1]))
        F_group_old = F_group_new
        # F_group = update_F(F_group)
        num_iter += 1
    return F_group_matrix, num_iter
#calculate all groups using F_group and reward
def calc_all_groups(reward_group, F_group):
    # update order: F_old -> g -> v -> lambda -> ro -> F_new
    g_group = solve_g_group(F_group, reward_group)
    v_group = solve_v_group(g_group)
    lambda_group = solve_lambda_group(g_group, v_group, ic.maximizer)
    ro_group = solve_ro_group(lambda_group)
    return (reward_group, F_group, g_group, v_group, lambda_group, ro_group)

#given inital reward functions and 0 <= alpha <= 1, returns the time until 100*alpha% agents has reached final stage
def comple_time(reward_group, alpha=ic.alpha, F_initals=ic.F_group_old, tol=ic.tol, t_vec=ic.t_vec):
    F_eq = calc_F_eq(reward_group, F_initials=F_initals, tol=tol)
    return quantile(F_eq[-1], t_vec, alpha)

#plot the reward function, optimal effort and cdf of completion times in a 2*2 figure
def output_plots(reward_group, effort_group, cdf_group):
    import matplotlib.pyplot as plt
    height, width = 2, 2
    fig = plt.figure()
    # first graph are reward functions
    plt.subplot(height, width, 1)
    reward_labels = ["reward_{}".format(i) for i in ic.stage_range]
    plot_group("reward_functions", reward_group, ic.colors, labels=reward_labels, t_vec=ic.p_vec)
    # second graph are initial cdf Fs'
    plt.subplot(height, width, 2)
    effort_labels = ["effort_{}".format(i + 1) for i in ic.state_range]
    plot_group("Optimal efforts".format(iter), effort_group, colors=ic.colors, labels=effort_labels)
    # third graph are F_eqs'
    plt.subplot(height, width, 3)
    F_eq_labels = ["F_eq_{}".format(i) for i in ic.stage_range]
    plot_group("Equilibrium Cdf of completion time", cdf_group, colors=ic.colors, labels=F_eq_labels)
    return fig
    # plt.close(fig)

#plot the reward functions in a 3*2 figure, on the left are equal rewards, on the right are lump-sum rewards
def output_plots_eq_and_ls(eq_reward_group, ls_reward_group, effort_group, cdf_group):
    import matplotlib.pyplot as plt
    height, width = 2, 3
    fig = plt.figure()
    #equal reward part
    plt.subplot(height, width, 1)
    reward_labels1 = ["equal_reward_{}".format(i) for i in ic.stage_range]
    plot_group("reward_functions", eq_reward_group, ic.colors, labels=reward_labels1, t_vec=ic.p_vec)
    # second graph are initial cdf Fs'
    plt.subplot(height, width, 2)
    effort_labels = ["effort_{}".format(i + 1) for i in ic.state_range]
    plot_group("Optimal efforts".format(iter), effort_group, colors=ic.colors, labels=effort_labels)
    # third graph are F_eqs'
    plt.subplot(height, width, 3)
    F_eq_labels = ["F_eq_{}".format(i) for i in ic.stage_range]
    plot_group("Equilibrium Cdf of completion time", cdf_group, colors=ic.colors, labels=F_eq_labels)
    #lump-sum reward part
    plt.subplot(height, width, 4)
    reward_labels1 = ["equal_reward_{}".format(i) for i in ic.stage_range]
    plot_group("reward_functions", ls_reward_group, ic.colors, labels=reward_labels1, t_vec=ic.p_vec)
    # second graph are initial cdf Fs'
    plt.subplot(height, width, 5)
    effort_labels = ["effort_{}".format(i + 1) for i in ic.state_range]
    plot_group("Optimal efforts".format(iter), effort_group, colors=ic.colors, labels=effort_labels)
    # third graph are F_eqs'
    plt.subplot(height, width, 6)
    F_eq_labels = ["F_eq_{}".format(i) for i in ic.stage_range]
    plot_group("Equilibrium Cdf of completion time", cdf_group, colors=ic.colors, labels=F_eq_labels)

    # plt.close(fig)

'''4. test functions'''
#do a single test with specific (B,i,beta) values, can choose to output result to excel or plot the graphs
def single_test_print_plot(reward_group, output_address="OutPut/WhereamI.csv", B=0, beta=0, i=0, alpha_list=ic.alpha_list,print_option=True, plot_option=True):
    #simulate eq and print result to excel
    import csv
    if not(print_option or plot_option): return None
    #calculate equilibrium and optimal effort
    F_eq_matrix, num_iter = calc_F_eq(reward_group)
    F_eq = F_eq_matrix[-1]
    g_eq = solve_g_group(F_eq, reward_group)
    v_eq = solve_v_group(g_eq)
    lambda_eq = solve_lambda_group(g_eq, v_eq, ic.maximizer)

    # print to the screen
    print('B = {:3.2f}, T = {:3.2f}, i = {:3.2f}, beta = {:3.2f}'.format(B, ic.T, i, beta))
    for alpha in alpha_list:
        T_alpha = quantile(F_eq[-1], ic.t_vec, alpha)
        print("alpha = {:3.2f}, completion time = {}".format(alpha, T_alpha))

    #output the info if print_option is True
    if print_option:
        with open(output_address, 'a', newline='') as test_f:
            test_writer = csv.writer(test_f)
            test_writer.writerow([])
            test_writer.writerow(
                ['B = {}, T = {}, i = {}, beta = {}'.format(B, ic.T, i, beta)] + ['beta \ alpha'] + alpha_list
            )
            result_list = ["num iters = {}".format(num_iter), beta]
            for alpha in alpha_list:
                T_alpha = quantile(F_eq[-1], ic.t_vec, alpha)
                result_list.append(T_alpha)
            test_writer.writerow(result_list)
    #plot the graphs if plot_option is True
    if plot_option:
        output_plots(reward_group, lambda_eq, F_eq)
        plt.show()
        plt.savefig()

#for piecewise constant functions, compare lump-sum and equal rewards by plotting equilibrium distribution, optimal effort and reward function
#for the two scenarios
def single_test_print_plot2(reward_groups, param_tuple=(ic.budget, 10, 0.1), plot_option=True, plot_address="Output/Plots/WhereamI.pdf"):
    #simulate eq and print result to excel
    B, n, d = param_tuple
    # print to the screen
    print("B = {:3.2f}, n = {}, d = {}".format(B, n, d))

    if not plot_option: return None
    equal_reward_group, ls_reward_group = reward_groups
    all_groups = []
    for reward_group in reward_groups:
        #calculate equilibrium and optimal effort
        F_eq_matrix, num_iter = calc_F_eq(reward_group)
        F_eq = F_eq_matrix[-1]
        all_groups.append(calc_all_groups(reward_group, F_eq))
        print("\d")

    eq_effort_group, eq_F_group = all_groups[0][4], all_groups[0][1]
    ls_effort_group, ls_F_group = all_groups[1][4], all_groups[1][1]
    #plot the graphs if plot_option is True
    import matplotlib.pyplot as plt
    height, width = 2, 4
    fig = plt.figure()
    plt.subplot(height, width, 1)
    reward_labels = ["reward_{}".format(i) for i in ic.stage_range]
    plot_group("equal reward", equal_reward_group, ic.colors, labels=reward_labels, t_vec=ic.p_vec)
    plt.subplot(height, width, 2)
    effort_labels = ["effort_{}".format(i + 1) for i in ic.state_range]
    plot_group("optimal efforts".format(iter), eq_effort_group, colors=ic.colors, labels=effort_labels)
    plt.subplot(height, width, 3)
    F_eq_labels = ["F_eq_{}".format(i) for i in ic.stage_range]
    plot_group("completion time", eq_F_group, colors=ic.colors, labels=F_eq_labels)

    plt.subplot(height, width, 5)
    reward_labels = ["reward_{}".format(i) for i in ic.stage_range]
    plot_group("lump-sum reward", ls_reward_group, ic.colors, labels=reward_labels, t_vec=ic.p_vec)
    plt.subplot(height, width, 6)
    effort_labels = ["effort_{}".format(i + 1) for i in ic.state_range]
    plot_group("optimal efforts".format(iter), ls_effort_group, colors=ic.colors, labels=effort_labels)
    plt.subplot(height, width, 7)
    F_eq_labels = ["F_eq_{}".format(i) for i in ic.stage_range]
    plot_group("completion time", ls_F_group, colors=ic.colors, labels=F_eq_labels)

    #comparing the completion time distributions for equal vs. lump-sum rewards
    plt.subplot(height, width, 8)
    comp_times = [eq_F_group[-1], ls_F_group[-1]]
    test_labels = ["equal", "lump_sum"]
    plot_group("completion time for last stage", comp_times, ic.colors, test_labels)
    plt.savefig(plot_address)
    # plt.show()
    plt.close(fig)

#returns a list with indicated variable at the last position
#precondition: indicator contains one 1, n-1 0's
#e.g. original = [a,b,c,d,e], indicator = [0,1,0,0,0] -> [a,c,d,e,b]
def put_var_last(original, indicator=[0,1,0]):
    last = original[0]
    result = []
    for i in range(len(indicator)):
        if indicator[i] == 1: last = original[i]
        else: result.append(original[i])
    result.append(last)
    return result

    #for i in range(len)

#inserts last element to designated position according to indicator
#e.g. original = [a,b,c,d,e], indicator = [0,1,0,0,0] -> [a,e,b,c,d]
def put_var_back(result, indicator):
    from copy import deepcopy
    original = deepcopy(result)
    last = original.pop()
    for i in range(len(indicator)):
        if indicator[i]: original.insert(i, last)
    return original


#fix one of (B,i,beta) to be constant, change the other two; print result to excel
#precondition: one of (B,i,beta) is scalar, the other two are lists of values
#indicator indicates which variable is fixed, e.g. [0, 1, 0] means beta is fixed
def full_test_print(B_vals, beta_vals, i_vals, indicator, output_address,
                     var_name_list=ic.var_name_list, alpha_list=ic.alpha_list, reward_group_type="equal"):
    #put value and name of the fixed var at the last position
    ordered_names = put_var_last(["B", "beta", "i"], indicator)
    ordered_vals = put_var_last([B_vals, beta_vals, i_vals], indicator)
    val1_list, val2_list, fixed_val = ordered_vals[0], ordered_vals[1], ordered_vals[2]
    name1, name2, fixed_name = ordered_names[0], ordered_names[1], ordered_names[2]
    #fix the first element, loop through other elements and print result to excel
    import csv
    with open(output_address, 'a', newline='') as test1_f: #open the excel to write to
        test1_writer = csv.writer(test1_f)
        for var1 in val1_list:  # loop through first variable and print out a bunch of tables
            test1_writer.writerow([]) #new line
            test1_writer.writerow(["{}={},{}={},T={}".format(name1, var1, fixed_name, fixed_val, ic.T)])# the first row contains all var names
            test1_writer.writerow(["num of iters"] + ["{}/alpha".format(name2)] + alpha_list)

            for var2 in val2_list: # each data row
                # print out current values of all vars to screen
                print("{} = {:2.1f}, {} = {:2.1f}, {} = {:2.1f}".format(name1, var1, name2, var2, fixed_name, fixed_val))

                #reorder variables to define reward functions
                unordered_vars = put_var_back([var1, var2, fixed_val], indicator)
                budget, beta, i = unordered_vars[0], unordered_vars[1], unordered_vars[2]

                reward_func_type = "pw"
                # define reward function
                pw_reward = lambda p: ic.power_reward(budget, beta, i, p)
                pc_reward = lambda p: ic.piecewise_constant_reward(10, 0.1, p)
                reward_func = pw_reward if reward_func_type == "pw" else pc_reward

                # define reward group
                if reward_group_type == "lump_sum":
                    reward_group = ic.lump_sum_reward_group(reward_func, ic.num_stages)
                else:
                    reward_group = ic.equal_reward_group(reward_func, ic.num_stages)

                # implement fixpoint algo to calculate F-eq and num of iters
                F_matrix, num_iters = calc_F_eq(reward_group)
                #refer to cdf of last stage
                F_eq_group = F_matrix[-1]

                result_list = [num_iters, var2] #"num of iters/value of second var" is first cell of each data row

                for alpha in alpha_list:#each column (alpha)
                    T_alpha_var2 = quantile(F_eq_group[-1], ic.t_vec, alpha)  # calculate the 100*alpha% completion time
                    print("alpha = {:3.2f}, completion time = {:3.2f}".format(alpha, T_alpha_var2))# print out current alpha and completion time
                    result_list.append(T_alpha_var2)
                test1_writer.writerow(result_list)  # write the list as a row to csv file
            print() #newline
'''Formal algorithm part is done'''

'''other debug functions'''
#finite difference gradient function
#gradient self-written function
#precondtition: x_vec is evenly spaced; f(x[i]) = y[i] for all i
def self_grad_func(x_vec, y_vec):
    #use forward difference for grad[0]
    h = x_vec[1] - x_vec[0]
    grad_vec = [(y_vec[1] - y_vec[0])/h]
    n = len(y_vec)
    for i in range(1, n - 1):
        #use central difference for grad[i], for 1 <= i <= n-2
        grad_vec.append((y_vec[i+1] - y_vec[i-1])/(2*h) )
    #use backward difference for grad[n-1]
    grad_vec.append((y_vec[n - 1] - y_vec[n - 2])/h)
    return interp1d(x_vec, grad_vec, kind="cubic")

#second argument is a function
#calculates the gradient of y_func(x) at every value of x_vec
def self_grad_func2(x_vec, y_func):
    #use forward difference for grad[0]
    y_vec = [y_func(x) for x in x_vec]
    #works well if convextiy in [x_-1,x1] and [x_n-1,xn+1] doesn't change
    y_vec.insert(0, y_vec[0] - (y_vec[1]-y_vec[0])) #creates y_-1 to calculate y_0'
    y_vec.append(y_vec[-1] + (y_vec[-1] - y_vec[-2])) #creates y_n+1 to calculate y_n'
    h = x_vec[1] - x_vec[0]
    grad_vec = []
    n = len(x_vec)
    for i in range(1, n+1):
        #uses central difference for grad[i]=(y_vec[i+1]-y_vec[i-1])/2, for 0 <= i <= n-1
        grad_vec.append((y_vec[i+1] - y_vec[i-1])/(2*h) )
    return interp1d(x_vec, grad_vec, kind="cubic")
#gradient function behaves bad at endpoints but good in between

#test v_y(t), v_group contains x0+1 functions
def solve_vgroup_errors(g_group, v_group, t_vec = ic.t_vec):
    v_errors = []
    for state in range(0, num_stages): #for 0 <= y <= x0-1
        #v_y' + conv_conj(g_y+1 + v_y+1 - v_y) = 0
        grad_v_curr = self_grad_func2(t_vec, v_group[state])
        error_func = lambda t,state=state,grad_v_curr=grad_v_curr\
            : grad_v_curr(t) + ic.convex_conjugate(g_group[state](t)
                                        + v_group[state+1](t) - v_group[state](t) )
        v_errors.append(error_func)
        # v[x0] should be 0
    v_errors.append(lambda t: v_group[-1](t) - 0)
    return v_errors
#assume that lambda_y(t) is correctly implemented
#test ro_y(t)
def solve_rogroup_errors(lambda_group, ro_group, t_vec = ic.t_vec):
    #ro_0' = -lambda_0 * ro_0
    ro0_grad = self_grad_func2(t_vec, ro_group[0])
    ro0_error = lambda t: ro0_grad(t) + lambda_group[0](t)*ro_group[0](t)
    ro_errors = [ro0_error]
    #for 1 <= y <= x0, ro_y' = lambda_y-1 * ro_y-1 - lambda_y * ro_y
    for state in range(1, num_stages+1):
        roy_grad = self_grad_func2(t_vec, ro_group[state])
        roy_error = lambda t,state=state,roy_grad=roy_grad:\
            roy_grad(t) - ( lambda_group[state-1](t) * ro_group[state-1](t)
                            -lambda_group[state](t) * ro_group[state](t) )
        ro_errors.append(roy_error)
    return ro_errors
