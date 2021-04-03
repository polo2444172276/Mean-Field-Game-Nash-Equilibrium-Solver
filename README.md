# Mean Field Game Equilibrium Computation

Mean-field game theory is the study of strategic decision making by small interacting agents in very large populations. In continuous time a mean-field game is typically composed by a Hamilton–Jacobi–Bellman equation that describes the optimal control problem of an individual. Under fairly general assumptions it can be proved that a class of mean-field games is the limit as N &rarr; &infin;  of a N-player Nash equilibrium.

### Game Background
We analyze a specific type of mean-field game. Players gets rewarded for completing tasks(all players are the same and whatever type of task doesn't matter). There are multiple tasks/stages. The reward of each player is based on the rank (i.e. quantile of the completion time) among all players, and the rank-to-reward function is designed by the policy maker. Also, each player has a fixed cost budget that they can spend to boost up their efficiency i.e. shorten the completion time. 

### Fixed-Point Numerical Algorithm
This project implements the nash equilibrium (i.e. cumulative distribution function of completion time F<sub>T</sub>) solver with a fixed-point algorithm. The reward function is the input and can be changed by the user. The algorithm recursively updates the expected value function, unique pointwise maximizer, completion time inflow/outflow rate and F<sub>T</sub> by solving for a group of partial differential equations(PDE). The fixed-point algorithm coverges if the new F<sub>T</sub>  is close enough to the current F<sub>T</sub>.

### Tutorial
After setting up environment and packages, go to main() in single_test.py and modify the parameters of the reward function. There are two types of reward: piecewise 
reward and continuous, power function reward. Then run single_test.py module and the plots will be saved in /Outputs. 

### Example
The following figure is an example which contains piecewise reward function(left), optimal effort i.e. how each player allocates cost budget(middle), and completion time distribution F<sub>T</sub> (right) during each stage of the game.
![plot](https://github.com/polo2444172276/Mean-Field-Game-Nash-Equilibrium-Solver/blob/main/demo_picture.jpg?raw=true)
