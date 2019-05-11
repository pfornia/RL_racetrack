import helpers
import copy
import numpy as np
import random
import time

start = 'S'
goal = 'F'
wall = '#'

# Actions, default choices first
actions = [(0,0), (-1,0), (1,0),
            (0, -1), (0, 1), (-1,-1),
            (-1, 1), (1,-1), (1, 1)]

max_velocity = 5

vel_range = range(-max_velocity,max_velocity+1)

def calc_policy_from_Q(cols, rows, vel_range, Q, actions):
    """
    Takes the Q list, and returns the policy dictionary based on argmax of each state.
    """
    pi = {}
    for y in range(rows): 
        for x in range(cols):
            print("Calculating policy...", y, x)
            for vy in vel_range:
                for vx in vel_range:
                    pi[(y,x,vy,vx)] = actions[np.argmax(Q[y][x][vy][vx])]
            
    return(pi)

def nearest_open_cell(world, y_crash, x_crash, vy = 0, vx = 0, open = ['.', 'S', 'F']):
    """
    Find nearest (manhattan distance) open cell to handle crashes.
    Check crash site, then diamond of radius 1, then 2, etc., until find open cell
    
     E.g., 'diamond' of radius 2:
     
            .
           ...
          ..#..
           ... 
            .
   
    UPDATE: If velocity given, then only seach in the opposite direction.
        This will prevent many instances of "jumping" over walls
    """ 
    rows = len(world)
    cols = len(world[0])    
   
    #Adding ten ensures coverage even when flying off the map
    max_radius = rows + cols + 10

    #if vy == 0 and vx == 0:
    for radius in range(max_radius):
        if vy == 0: 
            y_off_range = range(-radius, radius + 1)
        elif vy < 0:
            y_off_range = range(0, radius + 1)
        else:
            y_off_range = range(-radius, 1)
        for y_offset in y_off_range:
            y = y_crash + y_offset
            x_radius = radius - abs(y_offset)
            if vx == 0:
                x_range = range(x_crash - x_radius, x_crash + x_radius + 1)
            elif vx < 0:
                x_range = range(x_crash, x_crash + x_radius + 1)
            else:
                x_range = range(x_crash - x_radius, x_crash + 1)
            for x in x_range:
                if y < 0 or y >= rows: continue
                if x < 0 or x >= cols: continue
                if world[y][x] in open: 
                    return(y,x)
        

    print("Something's wrong, no open squares found!")
    return

def calc_new_velocity(old_vel, accel, min_vel = -max_velocity, max_vel = max_velocity):
    """
    Return index of new velocity (in velocity list)
    """
    new_y = old_vel[0] + accel[0] 
    new_x = old_vel[1] + accel[1]
    if new_x < min_vel: new_x = min_vel
    if new_x > max_vel: new_x = max_vel
    if new_y < min_vel: new_y = min_vel
    if new_y > max_vel: new_y = max_vel
    
    return new_y, new_x

def calc_new_location(old_loc, vel, world):
    """
    Calculates a new location from old location and a velocity.
    """
    y,x = old_loc[0], old_loc[1]
    vy, vx = vel[0], vel[1]
    
    return y+vy, x+vx

def get_random_start(world):
    """
    Pick one of the starting values of the world, and return its coordinates.
    """
    starts = []
    for y,row in enumerate(world):
        for x,col in enumerate(row):
            if col == start:
                starts += [(y,x)]

    #return random.choice(starts)
    return starts[0]

def make_move(old_y, old_x, old_vy, old_vx, accel, world, deterministic = False, badcrash = False):
    """
    Get new state (location and velocity) from old state and an acceleration vector.

    Handles different crash scenarios, and returns state after the crash has been accounted for.
    """ 
    if not deterministic:
        #Non-determinism
        if random.random() > 0.8: 
            #print("Slipping!")
            accel = (0,0)
 
    rows = len(world)
    cols = len(world[0])    
    
    new_y, new_x, new_vy, new_vx = copy.deepcopy((old_y, old_x, old_vy, old_vx))
    new_vy, new_vx = calc_new_velocity((old_vy,old_vx), accel)
    temp_y, temp_x = calc_new_location((old_y,old_x), (new_vy, new_vx), world)

    new_y, new_x = nearest_open_cell(world, temp_y, temp_x, new_vy, new_vx)
    #If crash...
    if new_y != temp_y or new_x != temp_x:
        if badcrash and world[new_y][new_x] != 'F':
            new_y, new_x = get_random_start(world)
        new_vy, new_vx = 0,0

    return new_y, new_x, new_vy, new_vx

def val_iteration(world, badcrash = False, reward = 0,  gamma = 0.9, alpha = 0.25):
    """
    Returns a policy from a world. Uses the value iteration algorithm. See Sutton and Bartlow (Ch 4)
    """
    rows = len(world)
    cols = len(world[0])    

    #Check: Is this supposed to be random? Or is zero ok??
    #for all possible states s in S, V[s] = 0
    values = [[[[0 for _ in vel_range] for _ in vel_range] for _ in line] for line in world]
    
    #initialize Q
    Q = [[[[[0 for _ in actions] for _ in vel_range] for _ in vel_range] for _ in line] for line in world]
    
    #TODO: termination criteria!    
    for t in range(100):

        if t % 1 == 0: print("Iteration", t)
        values_last = copy.deepcopy(values)
        
        #for each s in S
        for y in range(rows):
            for x in range(cols):
                for vy in vel_range:        
                    for vx in vel_range:
                        if world[y][x] == wall:
                            values[y][x][vy][vx] = 0
                            continue
                
                        #for each a in A:    
                        for ai, a in enumerate(actions):
                            if world[y][x] == goal:
                                r = reward
                            else:
                                r = -1
  
                            new_y, new_x, new_vy, new_vx = make_move(
                                y, x, vy, vx, a, world, deterministic = True, badcrash = badcrash) 
                            action_value = values_last[new_y][new_x][new_vy][new_vx]
                            
                            new_y, new_x, new_vy, new_vx = make_move(
                                y, x, vy, vx, (0,0), world, deterministic = True, badcrash = badcrash) 
                            no_action_value = values_last[new_y][new_x][new_vy][new_vx]
                            
                            expected_value = 0.8*action_value + 0.2*no_action_value

                            Q[y][x][vy][vx][ai] = r + gamma*expected_value
                    
                        argMaxQ = np.argmax(Q[y][x][vy][vx])
                        values[y][x][vy][vx] = Q[y][x][vy][vx][argMaxQ]
             
        #Set all finish line tiles to reward
        for y in range(rows):
            for x in range(cols):
                if world[y][x] == goal:
                    for vy in vel_range:
                        for vx in vel_range:
                            values[y][x][vy][vx] = reward
        max_v_change = max([max([max([max([abs(values[y][x][vy][vx] - values_last[y][x][vy][vx]) for vx in vel_range]) for vy in vel_range]) for x in range(cols)]) for y in range(rows)])
        print("max change:", max_v_change)
        if max_v_change < 0.001:
             return(calc_policy_from_Q(cols, rows, vel_range, Q, actions))
             
    print("Quitting early after 100 iterations")
    return(calc_policy_from_Q(cols, rows, vel_range, Q, actions))

def q_learning(world, badcrash = False, reward = 0,  gamma = 0.9, alpha = 0.25, max_episodes = 500000):
    """ 
    Returns a policy from a world. Used the Q-learning algorithm. See Sutton and Bartlow(Ch 6)
    """
    rows = len(world)
    cols = len(world[0])    

    #initialize Q
    Q = [[[[[0 for _ in actions] for _ in vel_range] for _ in vel_range] for _ in line] for line in world]

    #max_episodes = 1000000
    max_episode_length = 1000

    for ep in range(max_episodes):
               
        if ep % 1000 == 0: print("Episode:", ep)
        #reset all goal states to reward value
        for y in range(rows):
            for x in range(cols): 
                if world[y][x] == goal: 
                    Q[y][x] = [[[reward for _ in actions] for _ in vel_range] for _ in vel_range] 
        
        #Choose random initial state
        y = np.random.choice(range(rows))
        x = np.random.choice(range(cols))
        vy = np.random.choice(vel_range) 
        vx = np.random.choice(vel_range) 

        for t in range(max_episode_length):
            if world[y][x] == goal: break
            if world[y][x] == wall: break

            a = np.argmax(Q[y][x][vy][vx])                        
            new_y, new_x, new_vy, new_vx = make_move(y, x, vy, vx, actions[a], world, badcrash = badcrash) 
            r = -1
            Q[y][x][vy][vx][a] = ((1 - alpha)*Q[y][x][vy][vx][a] + 
                alpha*(r + gamma*max(Q[new_y][new_x][new_vy][new_vx])))

            y, x, vy, vx = new_y, new_x, new_vy, new_vx

    print(Q)
    return(calc_policy_from_Q(cols, rows, vel_range, Q, actions))
