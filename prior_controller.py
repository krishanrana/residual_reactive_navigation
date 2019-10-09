#! /usr/bin/env python3
# Author: Krishan Rana
import numpy as np
import cv2
import random
import time
from matplotlib import pyplot as plt
import scipy.ndimage
from collections import deque

# Note that this controller is not perfect and we show that by learning a residual we can close the performance gap       
# All angles in this method have to be in robot frame and not global frame
class PriorController():

    def __init__(self):
        self.fov = 360

    def clipAngle(self, angle):
        #clip angle between 0 and FOV
        if angle < 0:
            angle = angle + self.fov
        elif angle > self.fov:
            angle = angle - self.fov

        return angle

    def attractiveField(self, angle_to_goal):
        mapTo360 = angle_to_goal + np.pi
        #init map with range of FOV 0 - FOV
        goal_bearing = self.clipAngle((np.rad2deg(mapTo360)))
        attraction_field = np.zeros([(self.fov+1)]) # 0 - FOV inculsive

        #set value of map to one at location of goal
        attraction_field[int(goal_bearing)] = 1

        #gradient is how sharp the attraction in the map is
        gradient = 1/(self.fov/2)

        #iterate through each angle in the fov map and compute linear relation to goal angle
        #ie compute ramp profile of map

        for angle in range(int(self.fov/2)):

            loc = int(self.clipAngle(goal_bearing - angle))
            attraction_field[loc] = 1 - angle * gradient

            loc = int(self.clipAngle(goal_bearing + angle))
            attraction_field[loc] = 1 - angle * gradient

        return attraction_field


    def repulsiveField(self, laser_scan):
        hit = np.flip((laser_scan < 0.5))
        struct = scipy.ndimage.generate_binary_structure(1, 1)
        hit = scipy.ndimage.binary_dilation(hit, structure=struct, iterations=20).astype(hit.dtype) 
        repulsive_field = np.zeros([(self.fov+1)])
        repulsive_field[int(self.fov/8) : int(7*self.fov/8)] = hit
        #repulsive_field[int(self.fov/4) : int(3*self.fov/4)] = hit
        return repulsive_field


    def computeResultant(self, dist_to_goal, angle_to_goal, laser_scan):

        Kw = 2.0
        Kv = 0.1
        att = self.attractiveField(angle_to_goal)
        rep = self.repulsiveField(laser_scan)

        result = att - rep
        peak = max(result)
        index = np.where(result==peak)[0][0]
        heading = np.deg2rad(index - self.fov/2)

        fov_map = np.arange(-self.fov/2, self.fov/2+1)

        # Compute a repulsive angular velocity to ensure robot steers away from obstacle
        rep_angle = self.fov/2 - np.where(laser_scan == np.min(laser_scan))[0][0]
        omega = -heading * Kw
        vel = (10 * Kv) * (1.0 - min(0.8 * abs(omega), 0.95)) # 10 instaead of distt-goal

        if np.min(laser_scan) < 0.4 and (50<np.where(laser_scan == np.min(laser_scan))[0][0]<220) :
            vel_rep = -1/np.min(laser_scan) * 0.01
            vel = vel + vel_rep
        else:
            vel_rep = 0

        omega = np.clip(omega, -1, 1)
        vel = np.clip(vel, -1, 1)

        return np.array([vel, omega])

