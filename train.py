from __future__ import division # so that 1/2 = 0.5 and not 0
import random
import copy
import numpy as np
import pandas as pd
import pickle
import sys

import helpers
import val_iter

#If inputs not provide, then prompt user.
if len(sys.argv) == 5:
    world_file = sys.argv[1]
    policy_file = sys.argv[2]
    selection = int(sys.argv[3])
    max_episodes = int(sys.argv[4])
else:
    world_file = input("Enter file path of track: ")
    policy_file = input("Enter file path of policy: ")
    print("Please select result of crash...")
    print("1. Stop car at edge of wall")
    print("2. Restart race at start")
    selection = int(input("Selection: "))
    max_episodes = int(input("Enter Number of episodes"))

#Read in ASCII world state (track).
world = helpers.read_world(world_file)

helpers.print_world(world)

if selection == 1:
    badcrash = False
else:
    badcrash = True

policy = val_iter.q_learning(world, badcrash = badcrash, max_episodes = max_episodes)

#Save policy for later simulation
with open(policy_file, 'wb') as handle:
    pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(policy)
