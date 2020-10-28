#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un, rand_norm
import numpy as np
import pickle
from tiles3 import tiles, IHT

memorySize = 4096
num_tilings = 8
alpha = 0.1/(num_tilings)
lamb = 0.9
epsilon = 0.0
w = []
z = []
discount = 1
iht = IHT(memorySize)

last_state = None
last_action = None

def my_tiles(state,action):
    (x,xdot) = state
    return tiles(iht,num_tilings,[8.0*x/(0.5+1.2),8.0*xdot/(0.07+0.07)],[action])

def q_hat(state,action):
    global w
    tiles = my_tiles(state,action)
    value = 0
    for t in tiles:
        value += w[t]
    return value

def max_q_hat(state):
    value = []
    for action in range(3):
        value.append(q_hat(state,action))
    np.array(value)
    return np.argmax(value)

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way
    global w
    w = [np.random.uniform(low = -0.001, high = 0.0) for x in range(memorySize)]

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global last_state
    global last_action
    global z

    prob = np.random.rand()
    if prob < epsilon:
        action = rand_in_range(3)
    else:
        action = max_q_hat(state)

    z = [0 for x in range(memorySize)]
    last_state = state
    last_action = action
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global w
    global z
    global last_state
    global last_action

    error = reward
    for t in my_tiles(last_state,last_action):
        error -= w[t]
        z[t] = 1

    prob = np.random.rand()
    if prob < epsilon:
        action = rand_in_range(3)
    else:
        action = max_q_hat(state)

    for t in my_tiles(state,action):
        error += discount*w[t]
    for i in range(memorySize):
        w[i] += alpha*error*z[i]
        z[i] = z[i]*discount*lamb
    last_state = state
    last_action = action
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w
    global z

    error = reward
    for t in my_tiles(last_state,last_action):
        error -= w[t]
        z[t] = 1
    for i in range(memorySize):
        w[i] += alpha*error*z[i]
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
            return w
    else:
        return "I don't know what to return!!"
