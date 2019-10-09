#! /usr/bin/env python3
# sRRN
# Author: Krishan Rana

import torch, torch.nn as nn
from prior_controller import *
from PointGoalNavigationEnv_v0 import *
from tensorboardX import SummaryWriter
import numpy as np
import os, copy, argparse, time, sys


PATH = os.path.dirname(os.path.realpath(__file__))
SEED = 10
EPISODES = 15

# Load the weights from a trained model
model_name = "1567049830.8576467_PointGoalNavigation_rrn_EnvType_1_"

timest = str(time.time())
log_dir = "runs/" + timest + "EVALUATION" + model_name
GAMMA = 0.99

def extract_network_uncertainty(state):
    action = pi(state.repeat(100,1))
    mean = torch.mean(action, dim=0)
    var = torch.var(action, dim=0)
    return mean, var

def load_weights():
    pi.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'pi.pth'))
    return

def prior_actor():
	dist_to_goal, angle_to_goal, _, _, laser_scan, _, _ = env._get_position_data()
	action = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
	return action

#------------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorNetwork, self).__init__()
        self.a1 = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU(), nn.Dropout(p=0.2), 
                                nn.Linear(400, 300), nn.ReLU(), nn.Dropout(p=0.2), 
                                nn.Linear(300, act_size), nn.Tanh())

    def forward(self, obs):
        return self.a1(obs)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

env = PointGoalNavigation(  num_beams    = 270,
              		        laser_range  = 0.5,
              		        laser_noise  = 0.01,
              		        angle_min    = -0.75*np.pi,
            		        angle_max    = 0.75*np.pi,
              		        timeout      = 300,
					        velocity_max = 5,
                            omega_max    = 10,
                            env_type     = 1,
                            reward_type  = "sparse")

prior = PriorController()

env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Load network weights
obs_size = env.observation_space.shape[0] + 2
act_size = env.action_space.shape[0]
pi = ActorNetwork (obs_size, act_size)
load_weights()

timestep = 0
success = 0
SPL_buffer = []
steps_buffer = []

for ep in range(EPISODES):
    done = False
    obs = env.reset()
    prior_action = prior_actor()
    obs = np.concatenate([prior_action, obs])
    obs_prev = obs

    timest = str(time.time())
    log_dir = "runs/" + timest + "EVALUATION" + model_name
    writer = SummaryWriter(log_dir=log_dir)
    
    timestep = 0
    steps = 0
    method_buffer = []
    
    while(not done):
        eps = np.random.random()
        policy_action, var = extract_network_uncertainty(torch.as_tensor(obs).float())
        policy_action = policy_action.detach().numpy()
        hybrid_action = (policy_action + prior_actor()).clip(env.action_space.low, env.action_space.high)
        prior_action = prior_actor()

        if eps > var[0] or eps > var[1]:
            print('Hybrid')
            nobs, rew, done, trail = env.step(hybrid_action)
            method_buffer.append('r')
            which_method = "hybrid"
        else:
            print('Prior')
            nobs, rew, done, trail = env.step(prior_action)
            method_buffer.append('p')
            policy_action[0] = 0
            policy_action[1] = 0
            which_method = "prior"
        nobs = np.concatenate([prior_actor(), nobs])

        act_string = "\n" + 'Step: ' + str(steps) + "\n" + 'Prior Action: ' + str(prior_action) + '\r'
        print(act_string, end='\r', flush=True)
        sys.stdout.flush()
        obs_prev = obs
        obs = nobs
        timestep += 1
        steps += 1
        env.render()


print('Done!')