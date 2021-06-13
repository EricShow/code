import os
import random
import math
from typing import List, Union
from tqdm import tqdm
import traceback

import numpy as np
import torch
def calculate_hint(points, c_pseudo):
    #distance = np.linalg.norm(points-c_pseudo[None,:],axis=1)
    distance = []
    for i in range(len(points)):
        distance.append(np.linalg.norm(points[i]-c_pseudo))
    distance = np.array(distance)
    distance = torch.from_numpy(distance[None,:])
    hint_ls = []
    if(len(points )==2):
        bevel_edge_distance = np.linalg.norm(points[0]-points[1])
        hint_cos = (distance[0]** 2 +distance[1 ]** 2 -bevel_edge_distance**2) /( 2 *distance[0] *distance[1])
        if (hint_cos < 0):
            for i in range(len(points)):
                hint_ls.append(points[len(points ) - i -1])
    elif(len(points) ==3):
        for idx, pos in enumerate(points):
            hint_cos = None
            for idx1, pos1 in enumerate(points):
                if(idx1 !=idx):
                    bevel_edge_distance = np.linalg.norm(pos-pos1)
                    fenzi = distance[0][idx]**2 + distance[0][idx1]**2 - bevel_edge_distance**2
                    cos_temp = (distance[0][idx]**2 + distance[0][idx1]**2 - bevel_edge_distance**2 ) / \
                                ( 2 *distance[0][idx] *distance[0][idx1])
                    print(cos_temp)
                    if hint_cos == None:
                        hint_cos = cos_temp
                        hint = pos1
                    else:
                        hint_cos = min(hint_cos, cos_temp)
                        if(hint_cos == cos_temp):
                            hint = pos1
            hint_ls.append(hint)
    return hint_ls
points = []
points.append([1,0,0.1])
points.append([0,1,0])
points.append([-1,0,0])
#print(len(points))
points = np.array(points)
print(points)
ct = []
ct.append([0,0,0])
ct = np.array(ct)
ct = torch.from_numpy(ct)
print(ct[None,:])
#print(points.size())
#print(ct.size())
#
#print("hint", hint)
s = torch.from_numpy(points)

hint = calculate_hint(s,ct)
print(s)
print(hint)