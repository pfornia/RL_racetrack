import numpy as np
import copy
import helpers
import time
import val_iter
import pickle
import random
import sys

def simulate_race(world, policy, badcrash = True, animate = False, start = 'S', goal = 'F', frame_time = 0.5):
    """
    Simulate a race from a prescribed policy. Determines average speed (performance) of a policy.

    input:
        world: list of lists containing world tiles (open vs wall)
        policy: dictionary prescribing suggested action for each state.
            Key is 4-tuple of y,x,vy,vx, and value is action in form of y,x acceleration.
        badcrash: boolean indicating restarting "bad" crash scenario.
        animate: If true, will print ASCII of map with current location of a new state every half second.
    
    Prints average time ot finish race.

    """
    world_display = copy.deepcopy(world)

    starting_coord = val_iter.get_random_start(world)

    y,x = starting_coord
    vy,vx = 0,0

    stop_clock = 0    

    for i in range(500):     
        world_display[y][x] = 'O'     
        if animate: 
            helpers.print_world(world_display)
            time.sleep(frame_time)
        world_display[y][x] = 'x'     
        
        a = policy[(y,x,vy,vx)]
        #print("location:", y,x)
        #print("v:", vy, vx)
        #print("a:", a)

        if world[y][x] == goal: return i 
        
        y,x,vy,vx = val_iter.make_move(y, x, vy, vx, a, world, badcrash = badcrash)

    
        if vy == 0 and vx == 0:
            stop_clock += 1
        else:
            stop_clock = 0

        if stop_clock == 5:
            print("Car stuck at %d,%d, simluation ending." % (y,x))
            return i

    print("Timeout after 500 steps.")
    return 500

#If inputs not provide, then prompt user.
if len(sys.argv) == 4:
    world_file = sys.argv[1]
    policy_file = sys.argv[2]
    selection = int(sys.argv[3])
else:
    world_file = input("Enter file path of track: ")
    policy_file = input("Enter file path of policy: ")
    print("Please select result of crash...")
    print("1. Stop car at edge of wall")
    print("2. Restart race at start")
    selection = int(input("Selection: ")) 


#Read in ASCII world state (track).
world = helpers.read_world(world_file)

with open(policy_file, 'rb') as handle:
    policy = pickle.load(handle)

races = 50
tot_steps = 0

if selection == 1:
    badcrash = False
else:
    badcrash = True

random.seed(3)
for _ in range(races):
    tot_steps += simulate_race(world, policy, badcrash = badcrash, animate = True)

print("Race completed after", tot_steps/races, "steps on avg.")
