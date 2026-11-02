# !/usr/bin/env python
# -*-coding:utf-8-*-
"""
DigiSun: a software to transform sunspot drawings into exploitable data. 
It allows to scan drawings, extract its information and store it in a database.
Copyright (C) 2019 Sabrina Bechet at Royal Observatory of Belgium (ROB)

This file is part of DigiSun.

DigiSun is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DigiSun is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DigiSun.  If not, see <https://www.gnu.org/licenses/>.
"""

import math
import numpy as np

def group_frame2(zurich, radius, latitude, longitude, center_x, center_y):
    """
    estimate the size of the frame side to contain a 
    group of a given zurich type and a given position.
    """
    
    #distance_from_center = math.sqrt((posx - center_x)**2 +
    #                                 (posy - center_y)**2)

    if latitude==0.0 and longitude==0.0:
        print('hey the positions are nul!')
        return int(radius/6.)
    
    #if distance_from_center < radius:
        #center_to_limb = (math.asin(distance_from_center *
        #                            1./radius))
        
    if zurich in ['A', 'J']:
        step_x = (radius/30.) * abs(math.cos(float(longitude)))
        step_y = (radius/30.) * abs(math.cos(float(latitude)))
    elif zurich in ['H']:
        step_x = (radius/18.) * abs(math.cos(float(longitude)))
        step_y = (radius/18.) * abs(math.cos(float(latitude)))
    elif zurich in ['B', 'C', 'D']:
        step_x = (radius/9.) * abs(math.cos(float(longitude)))
        step_y = (radius/9.) * abs(math.cos(float(latitude)))
    elif zurich in ['E']:
        step_x = (radius/6.) * abs(math.cos(float(longitude)))
        step_y = (radius/6.) * abs(math.cos(float(latitude)))
    elif zurich in ['F', 'G', 'X']:
        step_x = (radius/4.) * abs(math.cos(float(longitude)))
        step_y = (radius/4.) * abs(math.cos(float(latitude)))

    max_step = np.max([step_x, step_y])
    
    return int(step_x * 2), int(step_y * 2) 

def group_frame(zurich, radius, latitude, longitude, center_x, center_y):
    """
    estimate the size of the frame side to contain a 
    group of a given zurich type and a given position.
    """
    
    #distance_from_center = math.sqrt((posx - center_x)**2 +
    #                                 (posy - center_y)**2)

    if latitude==0.0 and longitude==0.0:
        print('hey the positions are nul!')
        return int(radius/6.)
    
    #if distance_from_center < radius:
        #center_to_limb = (math.asin(distance_from_center *
        #                            1./radius))
        
    if zurich in ['A', 'J']:
        step_x = (radius/30.) * abs(math.cos(float(longitude)))
        step_y = (radius/30.) * abs(math.cos(float(latitude)))
    elif zurich in ['H']:
        step_x = (radius/18.) * abs(math.cos(float(longitude)))
        step_y = (radius/18.) * abs(math.cos(float(latitude)))
    elif zurich in ['B', 'C', 'D']:
        step_x = (radius/9.) * abs(math.cos(float(longitude)))
        step_y = (radius/9.) * abs(math.cos(float(latitude)))
    elif zurich in ['E']:
        step_x = (radius/6.) * abs(math.cos(float(longitude)))
        step_y = (radius/6.) * abs(math.cos(float(latitude)))
    elif zurich in ['F', 'G', 'X']:
        step_x = (radius/4.) * abs(math.cos(float(longitude)))
        step_y = (radius/4.) * abs(math.cos(float(latitude)))

    max_step = np.max([step_x, step_y])
    
    return int(step_x * 2)

