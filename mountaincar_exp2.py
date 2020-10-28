#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

import numpy as np
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

def my_tiles(state,action):
    (x,xdot) = state
    return tiles(iht,num_tilings,[8.0*x/(0.5+1.2),8.0*xdot/(0.07+0.07)],[action])

def q_hat(indices,w):
    value = 0
    for t in indices:
        value += w[t]
    return value

if __name__ == "__main__":
    num_episodes = 1000
    num_runs = 1

    steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            print '\tepisode {}'.format(e+1)
            RL_episode(0)
            steps[r,e] = RL_num_steps()
    #np.save('steps',steps)
    w = RL_agent_message("ValueFunction")
    numActions = 3
    steps = 50
    fout = open('value','w')
    for i in range(steps):
        for j in range(steps):
            values = []
            for a in range(3):
                indices = tiles(iht,num_tilings,[8.0*i/(0.5+1.2),8.0*j/(0.07+0.07)],[a])
                values.append(-1*(q_hat(indices,w)))
            height = max(values)
            fout.write(repr(height) + ' ')
        fout.write('\n')
        fout.close()
