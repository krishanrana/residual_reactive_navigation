import numpy as np
import cv2
import random
import time
from matplotlib import pyplot as plt
import scipy.ndimage
from collections import deque

        
# All angles in this method have to be in robot frame and not global frame
# This should be more reactive than the force method

class PotentialFieldsController():

    def __init__(self):
        self.fov = 360
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True)
        self.buffer = deque(maxlen=2)
        self.buffer.append(0)
        self.buffer.append(0)

    def clipAngle(self, angle):
        #clip angle between 0 and FOV
        if angle < 0:
            angle = angle + self.fov
        elif angle > self.fov:
            angle = angle - self.fov

        return angle

    def attractiveField(self, angle_to_goal):
        mapTo360 = angle_to_goal + np.pi
        #print(np.rad2deg(angle_to_goal))
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

        # fov_map = np.arange(-self.fov/2, self.fov/2+1)
        # plt.plot(fov_map, attraction_field)
        # plt.show(block=False)
        # plt.pause(0.00000001)
        # plt.cla()

        return attraction_field


    def repulsiveField(self, laser_scan):
        hit = np.flip((laser_scan < 0.5))
        struct = scipy.ndimage.generate_binary_structure(1, 1)
        hit = scipy.ndimage.binary_dilation(hit, structure=struct, iterations=20).astype(hit.dtype) #30
        #hit = 1 - laser_scan*2
        repulsive_field = np.zeros([(self.fov+1)])
        #repulsive_field[int(self.fov/4) : int(3*self.fov/4)] = hit
        repulsive_field[int(self.fov/8) : int(7*self.fov/8)] = hit

        return repulsive_field


    def computeResultant(self, dist_to_goal, angle_to_goal, laser_scan):

        Kw = 2# 2
        Kv = 0.1 # 0.1
        att = self.attractiveField(angle_to_goal)
        rep = self.repulsiveField(laser_scan)

        result = att - rep
        peak = max(result)
        index = np.where(result==peak)[0][0]
        heading = np.deg2rad(index - self.fov/2)
        self.buffer.append(heading)
        if abs(self.buffer[0] - self.buffer[1]) > 10:
            heading = self.buffer[1]
        
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

        # if abs(rep_angle) < 45:
        #     omega_rep = -1/(rep_angle) * 5
        #     print('omega: ', omega)
        #     print('omega rep: ', omega_rep)
        #     omega = omega + omega_rep
        # else:
        #     omega_rep = 0


        omega = np.clip(omega, -1, 1)
        vel = np.clip(vel, -1, 1)

        #vel = 0
        # self.ax1.cla(), self.ax1.plot(fov_map, att), self.ax1.set_title('Attractor')
        # self.ax2.cla(), self.ax2.plot(fov_map, rep), self.ax2.set_title('Repulsor')
        # self.ax3.cla(), self.ax3.plot(fov_map, result), self.ax3.set_title('Resultant')

        # plt.show(block=False)
        # plt.pause(0.00000001)

        #print([vel, omega])

        return np.array([vel, omega])
        #return np.array([omega])

# Angle set to -45 to 45
# env = ToySubsumptionEnv(angle_min=-np.pi/4, angle_max=np.pi/4, num_beams=90, k=0.2, t=0.001, laser_noise = 0)
# obs = env.reset() 
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# nav = PotentialFieldsController()
# angle_tolerance = 3

# while(True):
#     env.render()
    
#     heading = nav.computeResultant(obs)
    
#     if abs(heading) < angle_tolerance   : action = 0
#     elif heading < -angle_tolerance     : action = 2
#     elif heading > angle_tolerance      : action = 1
    
#     obs, rew, done, _ = env.step(action)
    
#     if done:
#         obs = env.reset()

