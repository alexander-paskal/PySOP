"""
Contains algorithms for optimizing values and policies for Markov Decision Processes

A Markov Decision Process (MDP) is defined as a game between an agent and an environment with potentially
infinite, non-terminating paths and perfect information. It can be represented by 4 key components:

- states: a set of all viable states
- actions: a mapping of each state to its available actions
- Markovian Transition Model: a function accepting a state, an action and a subsequent state and returning the probability
of the subsequent state being reached from the given state after taking the given action. NOTE: since these are used to take
expectations for a given action, they will be modeled as nested mappings of the form:
    
    P[s][a][s_prime] = probability

or equally

    P[s][a] = {s_prime1: 0.5, s_prime2: 0.5}

- Reward: a function accepting a state and returning the reward for that state. 

An MDP aims to model the Values of each state, which are defined by the Bellman equation:

    V(s) = R(s) + gamma * MAX over all a (  SUM of all s_prime in children[a] ( P[s][a][s_prime] * V(s_prime) ) )

Once these values are obtained, the optimal policy at any state s can be defined as the action which yields the 
maximum expected value of all children s_prime, with probabilities defined by the Markovian Transition Model.

Two methods by which MDPs can be solved are Value Iteration and Policy Iteration.

Value Iteration is performed by applying the Bellman update to each state value, which done by initializing all values 
to some arbitrary number and setting the value of each state equal to the Bellman value at that state for all
states, and repeating until all values converge to a unique solution, with convergence being defined as the maximum state
difference being less than some threshold epsilon.
"""
import numpy as np
import random


def value_iteration(states, actions, transition, reward, discount=0.9, iv=0, epsilon=0.1):
    """
    Updates state values according to the bellman update, explained above. Repeat until convergence,
    defined as maximum change in value for a state over an iteration being less than epsilon
    :param states: a List of valid states in the MDP
    :param actions: a Dict mappings states to lists of possible actions from that state. Null values should be empty lists
    :param transition: a nested Dict of the form transition[s][a] = {s1: 0.5, s2: 0.2, s3: 0.3}
    :param reward: a function accepting a state and returning the reward for that state
    :param discount: Optional, a hyperparameter for the bellman equation governing speed of contraction.
    :param iv: Optional, initial values for the Bellman equation. Used only for visualization, update will converge regardless
    :param epsilon: Optional, convergence threshold
    :return: values, history: Dict mapping states to computed values, List of all values Dicts across iterations
    """
    # intialize values
    values = {
        state: iv for state in states
    }

    delta = 0
    history = [values]
    
    while True:
        new_values = {}
        for state in states:

            action_values = []
            for action in actions(state):
                action_value = 0
                for s_prime, prob in transition[state][action].items():
                    action_value += prob * values[s_prime]
                action_values.append(action_value)

            if len(action_values) > 0:
                optimal_action_value = max(action_values)
            else:
                optimal_action_value = 0

            new_value = reward(state) + discount * optimal_action_value
            abs_diff = abs(new_value - values[state])
            if abs_diff > delta:
                delta = abs_diff

            new_values[state] = new_value

        values = new_values
        history.append(values)
        if delta < epsilon:
            break
        delta = 0

    return values, history


def policy_iteration(states, actions, transition, reward, discount=0.9, iv=0, epsilon=0.1, seed=0):
    """
    Performs policy iteration on states to find the optimal policy
    :param states:
    :param actions:
    :param transition:
    :param reward:
    :param policy:
    :param discount:
    :param iv:
    :param epsilon:
    :return:
    """
    random.seed(seed)

    # initialize random policy
    policy = { k: [random.choice(v)] if len(v) > 0 else list() for k, v in actions.items()}

    while True:
        # solve for values at current policy
        values = value_iteration(states=states, actions=policy, transition=transition, reward=reward, discount=discount, iv=iv, epsilon=iv)

        unchanged = True
        for s in states:

            # evaluate policy action
            p_action = policy[s]
            p_value = 0
            for s_prime, prob in transition[s][p_action].items():
                p_value += values[s_prime] * prob

            # compare all possible actions to policy action
            for a in actions[s]:

                # evaluate action
                action = policy[s]
                value = 0
                for s_prime, prob in transition[s][a].items():
                    value += values[s_prime] * prob

                # compare and update policy
                if value > p_value:
                    policy[s] = a
                    p_value = value
                    unchanged = True
        if unchanged:
            break

    return policy
