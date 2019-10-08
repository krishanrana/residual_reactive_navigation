# Laser Based PointGoalNavigation Gym Environment
# Author: Jake Bruce, Krishan Rana

import numpy as np, cv2, sys, os, math, random, time
from gym import spaces
from Box2D import *
import os

PATH = os.path.dirname(os.path.realpath(__file__))

#========================================================
# HELPERS

def rnd(mn,mx): return np.random.random()*(mx-mn)+mn

#--------------------------------------------------------

class RaycastCallback(b2RayCastCallback):
    def __init__(self, **kwargs): super(RaycastCallback, self).__init__(**kwargs); self.hit = False; self.fixture = None; self.points  = []; self.normals = []
    def ReportFixture(self, fixture, point, normal, fraction): self.hit = True; self.fixture = fixture; self.points.append(point); self.normals.append(normal); return 1.0

#========================================================
# ENV CLASS

class PointGoalNavigation:
    def __init__(self, **kwargs):
        self.__dict__.update(dict(
            w = 2000, h = 1000, xmn = -2, xmx = 2, ymn = -1, ymx = 1,
            wait = 1, timeout = 1000, k = 0.1, t = 0.0005, eps = 0.025,
            angle_min = -0.75*np.pi, angle_max=0.75*np.pi,
            laser_range = 0.5, num_beams = 16, laser_noise = 0.01, velocity_max=1, omega_max=1, env_type=1, reward_type="sparse"))
        self.__dict__.update(kwargs)

        self.bg_img = cv2.resize(cv2.cvtColor(np.random.randint(225,256,(self.h//8,self.w//8)).astype(np.uint8), cv2.COLOR_GRAY2BGR), dsize=(self.w,self.h), interpolation=cv2.INTER_NEAREST)       
        self.action_space = spaces.Box(low=np.array([-self.velocity_max, -self.omega_max]), high=np.array([self.velocity_max, self.omega_max]), dtype=np.float64)
        self.num_laser_samples = self.num_beams
        self.num_bins = 15
        self.observation_space = spaces.Box(low=-1, high=1, shape=[self.num_bins+4])
        angle_span  = self.angle_max - self.angle_min
        angle_span += angle_span / self.num_laser_samples
        self.laser_angles = [self.angle_min+i/self.num_laser_samples*angle_span for i in range(self.num_laser_samples)]
        self._max_episode_steps = self.timeout
        self.collided = False
        self.goal_radius = 0.2
        self.pixels_per_meter = 500
        self.laser_obs = np.zeros(self.num_laser_samples)
        self.obs_loc = [-5, -5]
        self.actions_prev = [0, 0]
        self.done = False

#-----------------------------------------------------------------------------------------------------------------------------------------------#

    def w2p(self,x,y): return (int((x-self.xmn)/(self.xmx-self.xmn)*self.w), int(self.h-(y-self.ymn)/(self.ymx-self.ymn)*self.h))
    def w2r(self,r)  : return  int(r/(self.xmx-self.xmn)*self.w)

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def reset(self):
        self.world = b2World(gravity=(0,0))

        # outer walls
        wall_top      = self.world.CreateStaticBody(position=( 0, 1), shapes=b2PolygonShape(box=(2,0.025)))
        wall_bottom   = self.world.CreateStaticBody(position=( 0,-1), shapes=b2PolygonShape(box=(2,0.025)))
        wall_right    = self.world.CreateStaticBody(position=( 2, 0), shapes=b2PolygonShape(box=(0.025,2)))
        wall_left     = self.world.CreateStaticBody(position=(-2, 0), shapes=b2PolygonShape(box=(0.025,2)))
        self.outer_walls = [wall_top, wall_bottom, wall_right, wall_left]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
        # ENV 1

        if self.env_type == 1:
            self.barrier_block1 = self.world.CreateStaticBody(position=( 0, 0), shapes=b2PolygonShape(box=(0.1,0.1)))
            self.barrier_block2 = self.world.CreateStaticBody(position=( -0.5, 0.7), shapes=b2PolygonShape(box=(0.1,0.15)))
            self.barrier_block3 = self.world.CreateStaticBody(position=( -1.2, 0.0), shapes=b2PolygonShape(box=(0.2,0.05)))
            self.barrier_block4 = self.world.CreateStaticBody(position=( -1.35, 0.25), shapes=b2PolygonShape(box=(0.05,0.2)))
            self.barrier_block5 = self.world.CreateStaticBody(position=( -1, -0.5), shapes=b2PolygonShape(box=(0.08,0.15)))
            self.barrier_block6 = self.world.CreateStaticBody(position=( -0.1, -0.8), shapes=b2PolygonShape(box=(0.2,0.05)))
            self.barrier_block7 = self.world.CreateStaticBody(position=( 0.15, -0.65), shapes=b2PolygonShape(box=(0.05,0.2)))
            self.barrier_block8 = self.world.CreateStaticBody(position=( 1.2, 0.5), shapes=b2PolygonShape(box=(0.4,0.05)))
            self.barrier_block9 = self.world.CreateStaticBody(position=( 0.9, -0.5), shapes=b2PolygonShape(box=(0.2,0.3)))
            self.barrier_block10 = self.world.CreateStaticBody(position=( 0.3, 0.8), shapes=b2PolygonShape(box=(0.05,0.2)))
            self.barrier_walls = [self.barrier_block1, self.barrier_block2, self.barrier_block3, self.barrier_block4, self.barrier_block5, 
                                self.barrier_block6, self.barrier_block7, self.barrier_block8, self.barrier_block9, self.barrier_block10]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

        obstacle_mask = np.zeros([self.h, self.w])
        
        for wall in self.barrier_walls:
            points = []
            for local_point in wall.fixtures[0].shape.vertices:
                world_point = wall.GetWorldPoint(local_point)
                pix_point = self.w2p(*world_point)
                points.append(pix_point)
            pt1 = np.array(points[3])
            pt2 = np.array(points[1])
            # Dilate with radius of robot
            obstacle_mask[pt1[1]-25:pt2[1]+25, pt1[0]-25:pt2[0]+25] = 1

        # Search for obstacle free goal location
        while(1):
            goalx = rnd(0.5, 1.8)
            goaly = rnd(0.85, -0.85)
            goalpt = self.w2p(*np.array([goalx, goaly]))
            if obstacle_mask[goalpt[1], goalpt[0]] == 1:
                continue
            else:
                break
        self.goal = np.array([goalx, goaly])

        # Search for obstacle free robot location
        while(1):
            robotx = rnd(-1.8, -0.2)
            roboty = rnd(0.85, -0.85)
            robotpt = self.w2p(*np.array([robotx, roboty]))
            if obstacle_mask[robotpt[1], robotpt[0]] == 1:
                continue
            else:
                break
        self.robot_loc = np.array([robotx, roboty])

        # Initialise Agent
        self.agent_body  = self.world.CreateDynamicBody(position=(self.robot_loc[0], self.robot_loc[1]), angle=rnd(-np.pi,np.pi), angularVelocity=0, linearDamping=20.0, angularDamping=30.0)
        self.agent_shape = self.agent_body.CreateFixture(shape=b2CircleShape(pos=(0,0), radius=0.05), density=0.1, friction=0.3)
        self.agent_body.mass = 5

        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        agent_point = self.w2p(*agent_loc)

        self.timestep = 0

        return self._obs()

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def render(self):

        agent_color = (0,0,0)
        img = self.bg_img.copy()

        # draw goal
        radius = self.w2r(self.agent_shape.shape.radius) 
        cv2.circle(img, self.w2p(*self.goal), int(self.pixels_per_meter * self.goal_radius*0.9), (99,245,66), 45)
        cv2.circle(img, self.w2p(*self.goal), int(self.pixels_per_meter * self.goal_radius*0.35), (0,0,255), 40)

        # draw laser rays
        laser_img = img.copy()
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        points = [self.w2p(*agent_loc)]
        for angle, dist in zip(self.laser_angles, self.laser_obs):
            end = agent_loc + [dist*np.cos(self.agent_body.angle + angle), dist*np.sin(self.agent_body.angle + angle)]
            p2 = self.w2p(*end)
            points.append(p2)
        cv2.fillPoly(laser_img, pts=[np.array(points)], color=(128,0,128))
        cv2.polylines(laser_img, pts=[np.array(points)], isClosed=True, color=(255,0,255), thickness=3)
        for angle, dist in zip(self.laser_angles, self.laser_obs):
            end = agent_loc + [dist*np.cos(self.agent_body.angle + angle), dist*np.sin(self.agent_body.angle + angle)]
            p2 = self.w2p(*end)
            cv2.line(laser_img, points[0], p2, (255,0,255), 3)
        img = laser_img//4 + img//4*3 # blend laser image at a low alpha

        # draw walls
        for wall in self.outer_walls + self.barrier_walls:
            points = []
            for local_point in wall.fixtures[0].shape.vertices:
                world_point = wall.GetWorldPoint(local_point)
                pix_point = self.w2p(*world_point)
                points.append(pix_point)
            cv2.fillConvexPoly(img, points=np.array(points), color=(64,64,64))
            cv2.polylines(img, pts=[np.array(points)], isClosed=True, color=(0,0,0), thickness=8)

        # Agent position and shape data
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        pix_point = self.w2p(*agent_loc)
        radius    = self.w2r(self.agent_shape.shape.radius)

        # Draw agent
        cv2.circle(img, pix_point, radius, agent_color, -1)
       
        # draw orientation vector
        cv2.circle(img, self.w2p(*(np.array(agent_loc) + [0.060*np.cos(self.agent_body.angle), 0.060*np.sin(self.agent_body.angle)])), self.w2r(0.02), (50,50,50), -1)

        # show image
        cv2.namedWindow("PointGoalNavigation", cv2.WINDOW_NORMAL)
        cv2.imshow("PointGoalNavigation", img)
        key = cv2.waitKey(self.wait) & 0xff
        if   key ==       27: sys.exit(0)
        elif key == ord('r'): self.reset()
        elif key == ord(' '):
            key = cv2.waitKey(self.wait)&0xff
            while key != ord(' ') and key != 27: key = cv2.waitKey(self.wait)&0xff
            if key == 27: sys.exit(0)

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def goal_achieved(self):
        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        to_goal = self.goal - np.array(agent_loc)
        return np.linalg.norm(to_goal) < self.goal_radius

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def step(self, action):

        # Scale the actions by their maximums
        lin = float(action[0] * 5)
        omega = float(action[1] * 10)

        velocity = (lin*np.cos(self.agent_body.angle), lin*np.sin(self.agent_body.angle))
        
        self.agent_body.linearVelocity = (velocity)
        self.agent_body.angularVelocity = (omega)


        # Previous range to goal
        prev_dist = self._obs()[-2]
        self.actions_prev = action

        # simulate
        self.world.Step(1/60, 10, 10)
        self.world.ClearForces()

        dist = self._obs()[-2]

        
        self.timestep += 1

        agent_loc = self.agent_body.GetWorldPoint(self.agent_shape.shape.pos)
        pix_point = self.w2p(*agent_loc)


        # Defining the rewards
        dense_rew = ((prev_dist - dist) * 100) - 0.1

        if self.timestep > self.timeout: 
            self.done = True 
            if self.reward_type == "sparse":
                return self._obs(),   0.0, True,  {}
            elif self.reward_type == "dense":
                return self._obs(),   0.0, True,  {}

        elif self.goal_achieved(): 
            self.done = True
            if self.reward_type == "sparse": 
                return self._obs(),   1.0, True,  {}
            elif self.reward_type == "dense":
                return self._obs(),   10, True,  {}
        else: 
            self.done = False
            if self.reward_type == "sparse": 
                return self._obs(),  -0.01, False, {}
            elif self.reward_type == "dense":
                return self._obs(),  dense_rew, False, {}

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def _laser_rays(self):
        agent_loc = np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        return [(agent_loc, agent_loc + [self.laser_range*np.cos(self.agent_body.angle + angle), self.laser_range*np.sin(self.agent_body.angle + angle)]) for angle in self.laser_angles]

#-----------------------------------------------------------------------------------------------------------------------------------------------#


    def _get_position_data(self):
        robot_angle = np.arctan2(np.sin(self.agent_body.angle), np.cos(self.agent_body.angle))
        to_goal = self.goal - np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        angle_to_goal = robot_angle - np.arctan2(to_goal[1], to_goal[0])
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        dist_to_goal  = np.linalg.norm(to_goal)
        robot_loc = np.array(self.agent_body.GetWorldPoint(self.agent_shape.shape.pos))
        laser_scan = self.laser_obs

        return dist_to_goal, angle_to_goal, robot_loc, robot_angle, laser_scan, self.goal, self.obs_loc


    def _obs(self):
        # Generate laser scan
        laser_samples = np.zeros(self.num_laser_samples)
        for i,(start, end) in enumerate(self._laser_rays()):
            callback = RaycastCallback()
            self.world.RayCast(callback, start, end)
            laser_samples[i] = min([np.linalg.norm(start - point) for point in callback.points]) if len(callback.points) > 0 else self.laser_range
        num_hits = (laser_samples < self.laser_range).sum()
        self.laser_obs = laser_samples.copy()
        # Applying noise to laser scan
        self.laser_obs[laser_samples < self.laser_range] *= np.random.normal(1,self.laser_noise,num_hits)
        # Get global data
        dist_to_goal, angle_to_goal, _, _, _, _,_ = self._get_position_data()
        hit = (laser_samples < self.laser_range).astype(np.float)
        self.collided = np.any((laser_samples < 0.1).astype(np.float) == 1.0)

        laser_scan = np.zeros(self.num_bins)
        laser_hit = np.zeros(self.num_bins)
        div_factor = int(self.num_laser_samples/self.num_bins)
        for i in range(self.num_bins):
            laser_scan[i] = np.mean(self.laser_obs[i*div_factor:(i*div_factor+div_factor)]) 
            laser_hit[i] = (laser_scan[i] < self.laser_range).astype(np.float)
            
        return np.concatenate([laser_scan, # distances
                              #laser_hit, # hit bools
                              self.actions_prev,
                              [dist_to_goal], # linear distance to goal
                              [angle_to_goal]]) # angular difference to goal

    
    def seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.np_random.seed(seed)
        return
