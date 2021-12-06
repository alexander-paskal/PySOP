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

python >= 3.8
numpy >= 1.2
matplotlib >= 3.4

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

