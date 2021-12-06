"""
Contains basic reinforcement-learning algorithms

Reinforcement Learning comes into play when attempting to learn state values or
an optimal policy for an MDP in which we do not know know the environment i.e.
we don't know our Markovian Transition Model P(s' | s,a).

Without this, we cannot solve for optimal states explicitly, so we instead sample
and develop estimations of our expected rewards over time.

There are three main algorithms implemented here:

- Monte Carlo Policy Evaluation
- Temporal Difference Policy Evaluation
- Tabular Q-learning
"""
import random
import math


def montecarlo_policy_evaluation(episodes, states, reward, discount=0.95):
    """
    Performs Monte Carlo Policy Evaluation. Takes in a number of trajectories and
    develops state value estimates for all states over time by computing the average
    reward-to-go obtained at each state over n visits
    :param episodes: A container or generator of trajectories, each trajectory being a List of states
    :param states: The full container of possible states in the MDP
    :param reward: a function accepting a state as an argument and returning a numeric reward
    :param discount: a discount value, between 0 and 1
    :return: values, visits: Dict mapping states to value estimates based on passed-in episodes, Dict mapping states
    to number of visits over the course of the algorithms run
    """

    values = {}
    visits = {}
    sums = {}
    for s in states:
        values[s] = 0
        visits[s] = 0
        sums[s] = 0

    # create custom rtg function
    reward_to_go = _rtg_factory(reward)

    for episode in episodes:
        i = 0
        for s, reward in episode:
            sums[s] += reward_to_go(episode[i:], discount)
            visits[s] += 1
            values[s] = sums[s] / visits[s]
            i += 1

    return values, visits


def _rtg_factory(reward):
    def reward_to_go(trajectory, discount):
        """
        computes the reward-to-go for a given trajectory
        :param trajectory: List of states
        :param reward: function accepting state as argument and returning numeric reward
        :param gamma: discount factor, between 0 and 1
        :return:
        """
        rtg = 0
        for i, state in enumerate(trajectory):
            r = reward(state)
            rtg += discount ** i * r

        return rtg
    return reward_to_go


def temporal_difference_policy_evaluation(episodes, states, reward, alpha, discount=0.95):
    """
    Performs a temporal difference update on state value estimations by evaluating
    trajectory values and updating by weighted difference between current sample and
    previous estimation.

        V(s) <- V(s) + alpha ( R(s) + discount*V(s') - V(s))

    :param episodes:
    :param states:
    :param reward:
    :param alpha:
    :param discount:
    :return: values, visits: Dict mapping states to value estimates based on passed-in episodes, Dict mapping states
    to number of visits over the course of the algorithms run
    """
    # initialize values
    values = {}
    visits = {}
    for s in states:
        values[s] = 0
        visits[s] = 0


    for episode in episodes:
        episode.append(None)  # for while loop implementation

        i = 0
        s = episode[i]
        r = reward(s)
        while s is not None:
            next_s = episode[i+1]
            if next_s is None:  # if True, then will update td by alpha(current reward - current estimate)
                next_td = 0
                next_r = None
            else:
                next_r = reward(next_s)
                next_td = values[next_s]

            # performs update
            alp = alpha(visits[s] + 1)  # alpha is a callable accepting number of visits as a parameter
            td = values[s]
            result = td + alp*(r + discount*next_td - td)  #
            values[s] = result
            visits[s] += 1
            i += 1
            s, r = next_s, next_r

    return values, visits


def tabular_q_learning(episodes, states, actions, reward, alpha, discount=0.95, epsilon=0.4, seed=0):
    """
    Performs epsilon-greedy q-learning. Accepts a number of episodes over which to perform learning,
    updates Q-values for every state-action pair based on results of training.
    :param episodes:
    :param states:
    :param actions:
    :param reward:
    :param alpha:
    :param discount:
    :param epsilon:
    :param seed:
    :return:
    """
    random.seed(seed)

    # initialize values
    values = {}
    visits = {}
    for s in states:
        values[s] = {a: 0 for a in actions[s]}
        visits[s] = 0

    for episode in episodes:
        episode.append(None)  # helps with while loop implementation

        i = 0
        s = episode[i]
        r = reward(s)
        while s is not None:
            action = pick_action(s, values, epsilon)  # pick an action - epsilon-greedy
            next_s = episode[i + 1]
            if next_s is None:  # if True, then will update td by alpha(current reward - current estimate)
                next_q = 0
                next_r = None
            else:
                next_r = reward(next_s)
                next_q = values[next_s]

            # performs q-value update
            alp = alpha(visits[s] + 1)  # alpha is a callable accepting number of visits as a parameter
            q = values[s][action]
            result = q + alp * (r + discount * next_q - q)  #
            values[s] = result
            visits[s] += 1
            i += 1
            s, r = next_s, next_r

    return values, visits


def pick_action(s, values, epsilon):
    """
    Chooses an action for s based on an epsilon greedy strategy
    :param s: the state being evaluated
    :param values: The Q-values for all s-a pairs, nest Dict
    :param epsilon: the threshold for random choice, governing exploration vs. exploitation
    :return:
    """
    if random.random() < epsilon:
        return random.choice(values[s].keys())

    max_q_val = -math.inf
    max_action = None

    for action, value in values[s].items():
        if max_q_val < value:
            max_q_val = value
            max_action = action

    return max_action