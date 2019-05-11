from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd

def read_world( filename):
    with open( filename, 'r') as f:
        world_data = [x for x in f.readlines()]
    f.closed
    world = []
    for i,line in enumerate(world_data):
        if i > 0:
            line = line.strip()
            if line == "": continue
            world.append([x for x in line])
    return world

def print_world(world):
    print()
    for w in world: 
        text = ""
        for cell in w: text += cell
        print(text)
    print()
