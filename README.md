# PySOP

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gradient_descent_bowl.png)

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/search_gradient_dropwave.png)

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gridworld_a_star.png)

Here is a collection of Python implementations of various search and optimization algorithms. The main purpose of the repo is to:

- build the algorithms from scratch using numpy or native python
- visualize the results of the algorithms to develop intuitions about how they work and where to apply them
- provide an open-source api for accessing these algorithms
- provide implementations of common games and AIs to play against

# Installation

- python >= 3.8
- numpy >= 1.2
- matplotlib >= 3.4

You can run the following command from the terminal to install:

    >>> pip install -r requirements.txt
    
# How to Use

1. clone this repo
2. install all requirements
3. import from pysop.algorithms
4. read the documentation of the algorithms in question to understand their interface

# Algorithms

The following algorithms have implementations:

    - gradient descent
    - newtonian descent
    - simulated annealing
    - crossentropy search
    - search gradient
    - Djikstra's algorithm
    - A* Search
    - minimax
    - value iteration
    - policy iteration
    - monte carlo policy evaluation
    - temporal difference policy evaluation
    - tabular Q-learning
 
The following algorithms are planned to be added:

    - conjugate gradient descent
    - momentum-based gradient descent
    - adam optimizer
    - multi-layer perceptron
    - RRT, RRT*
    - D*
    - multiplayer minimax
    - alpha-beta pruning
    - deep Q-learning
    - policy gradient
    - PPO
    - MCTS
    - DPLL
    - CDCL


# Examples

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/newton_descent_ellipse.png)

Newtonian Descent for the fastest path to the global minimum of a convex function.

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gradient_descent_ellipse_alpha=0.1.png)
![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gradient_descent_ellipse_alpha=0.01.png)

The effect of the learning rate alpha on Gradient Descent

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/simulated_annealing_dropwave.png)
![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/1c-2-1.png)
![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/1c-2-5.png)

Simulated Annealing stochastic search finds the global minimum by randomly sampling and accepting some percentage of suboptimal points to get out of local minimums.

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/ce-dist2.png)
![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/ce-dist3.png)
![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/ce-dist5.png)

Crossentropy Search generates samples from a distribution, chooses a subsection of the most minimizing samples, fits the new distribution to those samples, and repeats.

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/search_gradient_dropwave.png)

Search Gradient computes the derivative of the function with respect to a distribution and performs gradient descent on the distribution parameters.

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gridworld_djikstra.png)

Djikstra's algorithm explores nodes with the next lowest cost in a frontier set, expanding outwords from a start point.

![alt text](https://github.com/alexander-paskal/PySOP/blob/main/visualizations/images/gridworld_a_star.png)

A* expands outwards but ranks potential nodes by both their cost and a heuristic function (manhattan distance from the goal node, in the case of gridworld). 


