#! /usr/bin/env python3
# Author: Vibha Dasagi, Krishan Rana

from __future__ import print_function, division
import numpy as np, torch, torch.nn as nn, sys, gym, copy, random, collections, tensorboardX
from termcolor import colored
from PointGoalNavigationEnv_v0 import *
from prior_controller import *
import os
import argparse


parser = argparse.ArgumentParser(description='Parameters for training.')

parser.add_argument('--method', type=str, default="residual", help="options include: policy, residual, combined")
parser.add_argument('--env_type', type=int, default=1)
parser.add_argument('--seed', type=int, default=14)
parser.add_argument('--viz_train', type=int, default=0)
parser.add_argument('--viz_eval', type=int, default=0)
parser.add_argument('--eval_delay', type=int, default=200)
parser.add_argument('--reward_type', type=str, default="sparse")
args = parser.parse_args()

#==============================================================================
# PARAMETERS

ENV           = 'PointGoalNavigation'
EPISODES      = int(2.5e3)
LR            = 1e-3
TRAIN_STEPS   = 1
SIG_ACT       = 0.2
SIG_TRAIN     = 0.3
EXPLORE_STEPS = 5000
BATCH         = 100
C             = 0.5
D             = 2               # Actor update frequency
TAU           = 5e-3
GAMMA         = 0.99
VIZ_STEP      = int(10000)
VIZ           = bool(args.viz_train)
VIZ_EVAL      = bool(args.viz_eval)
EVAL_FREQ     = 10
EVAL_DELAY    = args.eval_delay
SEED          = args.seed
LAMBDA_DECAY  = 0.99995
PATH          = os.path.dirname(os.path.realpath(__file__))
METHOD        = args.method
ENV_TYPE      = args.env_type
REWARD_TYPE   = args.reward_type

print('Method: ' + str(METHOD))
print('Env Type: ' + str(ENV_TYPE))
print('Reward Type: ' + str(REWARD_TYPE))

#==============================================================================
# SETUP

#env = gym.make(ENV)
env = PointGoalNavigation(  num_beams    = 270,
              		        laser_range  = 0.5,
              		        laser_noise  = 0.01,
              		        angle_min    = 3*-np.pi/4,
            		        angle_max    = 3*np.pi/4,
              		        timeout      = 300,
					        velocity_max = 5,
              		        omega_max    = 10,
                            env_type     = ENV_TYPE,
                            reward_type  = REWARD_TYPE)

prior = PriorController()

env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if METHOD == "residual":
    obs_size = env.observation_space.shape[0] + 2
elif METHOD == "policy":
    obs_size = env.observation_space.shape[0]
elif METHOD == "combined":
    obs_size = env.observation_space.shape[0]
elif METHOD == "prior":
    obs_size = env.observation_space.shape[0]
    
act_size = env.action_space.shape[0]

Exp = collections.namedtuple('Exp', ('obs', 'act', 'rew', 'nobs', 'done'))

timestep  = 0
lambda_    = 1
buf       = []
time_tag = str(time.time())
log_dir = "runs/" + time_tag + "_" + ENV + "_EnvType_" + str(ENV_TYPE) + "_RewardType_" + REWARD_TYPE 
model_name = time_tag + "_" + ENV + "_EnvType_" + str(ENV_TYPE) + "_RewardType_" + REWARD_TYPE
os.mkdir("pytorch_models/TD3_Data/" + model_name)
writer    = tensorboardX.SummaryWriter(log_dir=log_dir)

#==============================================================================
# MODEL

class CriticNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super(CriticNetwork, self).__init__()
        self.c1 = nn.Sequential(nn.Linear(obs_size+act_size, 400), nn.ReLU())
        self.c2 = nn.Sequential(nn.Linear(act_size+400, 300), nn.ReLU(), nn.Linear(300, 1))

    def forward(self, obs, act):
        x = self.c1(torch.cat([obs, act], dim=1))
        return self.c2(torch.cat([x, act], dim=1))

#------------------------------------------------------------------------------

if METHOD == "residual":
    class ActorNetwork(nn.Module):
        def __init__(self, obs_size, act_size):
            super(ActorNetwork, self).__init__()
            self.a1 = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU(), nn.Dropout(p=0.2),
                                    nn.Linear(400, 300), nn.ReLU(), nn.Dropout(p=0.2),
                                    nn.Linear(300, act_size), nn.Tanh())

        def forward(self, obs):
            return self.a1(obs)

else:
    class ActorNetwork(nn.Module):
        def __init__(self, obs_size, act_size):
            super(ActorNetwork, self).__init__()
            self.a1 = nn.Sequential(nn.Linear(obs_size, 400), nn.ReLU(),
                                    nn.Linear(400, 300), nn.ReLU(),
                                    nn.Linear(300, act_size), nn.Tanh())

        def forward(self, obs):
            return self.a1(obs)

#------------------------------------------------------------------------------

q1 = CriticNetwork(obs_size, act_size).cuda()
q2 = CriticNetwork(obs_size, act_size).cuda()
pi = ActorNetwork (obs_size, act_size).cuda()

opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
opt_a = torch.optim.Adam(pi.parameters(), lr=LR)

tq1 = copy.deepcopy(q1).cuda()
tq2 = copy.deepcopy(q2).cuda()
tpi = copy.deepcopy(pi).cuda()

#==============================================================================
# UTILITY FUNCTIONS

def save_weights():
    torch.save(q1.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' + model_name + 'q1.pth')
    torch.save(q2.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' + model_name + 'q2.pth')
    torch.save(pi.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' + model_name + 'pi.pth')
    torch.save(tq1.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tq1.pth')
    torch.save(tq2.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tq2.pth')
    torch.save(tpi.state_dict(), PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tpi.pth')
    return

def load_weights():
    q1.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'q1.pth'))
    q2.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'q2.pth'))
    pi.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'pi.pth'))
    tq1.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tq1.pth'))
    tq2.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tq2.pth'))
    tpi.load_state_dict(torch.load(PATH + '/pytorch_models/TD3_Data/' + model_name + '/' +  model_name + 'tpi.pth'))
    return

def prior_actor():
	#action = controller.p_controller_v2(dist_to_goal, angle_to_goal, laser_scan)
    dist_to_goal, angle_to_goal, robot_loc, robot_angle, laser_scan, goal_loc, obs_loc = env._get_position_data()
    action = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
    return action

def select_action(obs):
    global lambda_
    global timestep
    lambda_ *= LAMBDA_DECAY
    writer.add_scalar('{}/lambda'.format(ENV), lambda_, timestep)

    if timestep < EXPLORE_STEPS: 
        prior_action = prior_actor()
        policy_action = np.clip(env.action_space.sample(), env.action_space.low, env.action_space.high)
        combined_action = ((1/(1+lambda_))*policy_action) + ((lambda_/(1+lambda_))*prior_action)
        combined_action = np.clip(combined_action, env.action_space.low, env.action_space.high)
        residual_action = policy_action + prior_action 
        residual_action = np.clip(residual_action, env.action_space.low, env.action_space.high)


        return combined_action, policy_action, prior_action, residual_action

    policy_action = pi(torch.as_tensor(obs).float().cuda()).cpu().detach().numpy()
    policy_action = np.clip(policy_action + np.random.normal(0, SIG_ACT), env.action_space.low, env.action_space.high)

    prior_action = prior_actor()
    combined_action = ((1/(1+lambda_))*policy_action) + ((lambda_/(1+lambda_))*prior_action)
    combined_action = np.clip(combined_action, env.action_space.low, env.action_space.high)
    residual_action = policy_action + prior_action
    residual_action = np.clip(residual_action, env.action_space.low, env.action_space.high)

    return combined_action, policy_action, prior_action, residual_action

def clip_tensor(x, mn, mx):
    clipped = torch.max(torch.min(x, mx), mn)
    return clipped


def _extract_network_uncertainty(self, inputs):

    action = pi(inputs.repeat(100,1).cuda()).cpu()
    mean = torch.mean(action, dim=0)
    var = torch.var(action, dim=0)
    
    return mean, var


def evaluate_policy(eval_episodes=10, episode_num=0):
    print('Evaluate Policy...')
    avg_reward = 0.0
    avg_length = 0.0
    for _ in range(eval_episodes):

        if METHOD == "residual":
            prior_action = prior_actor()
            obs = np.concatenate([prior_action, env.reset()])
        elif METHOD == "combined":
            obs = env.reset()
        elif METHOD == "policy":
            obs = env.reset()
        elif METHOD == "prior":
            obs = env.reset()

        done = False
        while not done:
            prior_action = prior_actor()
            policy_action = pi(torch.as_tensor(obs).float().cuda()).cpu().detach().numpy()
            residual_action = (policy_action + prior_action).clip(env.action_space.low, env.action_space.high)
            
            if METHOD == "residual":
                nobs, reward, done, _ = env.step(residual_action)
                prior_action = prior_actor()
                nobs = np.concatenate([prior_action, nobs])
            elif METHOD == "policy":
                nobs, reward, done, _ = env.step(policy_action)
            elif METHOD == "combined":
                nobs, reward, done, _ = env.step(policy_action)
            elif METHOD == "prior":
                nobs, reward, done, _ = env.step(prior_action)
                nobs, reward, done, _ = env.step(random_action)

            avg_reward += reward
            avg_length += 1
            obs = nobs
            if VIZ_EVAL: env.render()

    avg_reward /= eval_episodes
    avg_length /= eval_episodes
    writer.add_scalar('{}/rewards_evaluation'.format(ENV), avg_reward, episode_num)
    writer.add_scalar('{}/length_evaluation'.format(ENV), avg_length, episode_num)
    save_weights()

    print('Training...')

    return avg_reward

#------------------------------------------------------------------------------

def episode():
    global timestep
    if METHOD == "residual":
        prior_action = prior_actor()
        obs = np.concatenate([prior_action, env.reset()])
    elif METHOD == "combined":
        obs = env.reset()
    elif METHOD == "policy":
        obs = env.reset()

    done = False
    rewards = 0
    length = 0
    while not done:
        
        combined_action, policy_action, prior_action, residual_action = select_action(obs)

        if METHOD == "residual":
            nobs, rew, done, _ = env.step(residual_action)
            prior_action = prior_actor()
            nobs = np.concatenate([prior_action, nobs])
            buf.append(Exp(obs, policy_action, rew, nobs, done))
        elif METHOD == "combined":
            nobs, rew, done, _ = env.step(combined_action)
            buf.append(Exp(obs, combined_action, rew, nobs, done))
        elif METHOD == "policy":
            nobs, rew, done, _ = env.step(policy_action)
            buf.append(Exp(obs, policy_action, rew, nobs, done))

        rewards += rew
        
        for step in range(TRAIN_STEPS):
            losses = train(buf)

        for k,v in losses.items():
            writer.add_scalar('{}/{}'.format(ENV,k), v, timestep)

        writer.add_scalar('{}/policy_velocity'.format(ENV), policy_action[0], timestep )
        writer.add_scalar('{}/policy_omega'.format(ENV), policy_action[1], timestep)
        writer.add_scalar('{}/prior_velocity'.format(ENV), prior_action[0], timestep )
        writer.add_scalar('{}/prior_omega'.format(ENV), prior_action[1], timestep)
        writer.add_scalar('{}/combined_velocity'.format(ENV), combined_action[0], timestep )
        writer.add_scalar('{}/combined_omega'.format(ENV), combined_action[1], timestep)
        if VIZ and (timestep > VIZ_STEP): env.render()


        obs = nobs.copy()
        timestep += 1
        length += 1

    return rewards, length

#------------------------------------------------------------------------------

def train(buf):
    if timestep < EXPLORE_STEPS: return {}
    batch = random.sample(buf, BATCH)
    batch = Exp(*map(lambda x: torch.FloatTensor(x).view(BATCH, -1).cuda(), zip(*batch)))

    nact  = tpi(batch.nobs)
    noise = torch.clamp(torch.randn(BATCH, act_size)*SIG_TRAIN, -C, C).detach().cuda()

    nact = clip_tensor((nact+noise), torch.Tensor(env.action_space.low).cuda(), torch.Tensor(env.action_space.high).cuda())
    #nact  = np.clip(nact + noise, env.action_space.low, env.action_space.high)
    #nact  = torch.clamp(nact + noise, -1.0, 1.0)

    qn1 = tq1(batch.nobs, nact)
    qn2 = tq2(batch.nobs, nact)
    qn  = torch.min(qn1, qn2)

    qs1 = q1(batch.obs, batch.act)
    qs2 = q2(batch.obs, batch.act)

    # train critic
    target  = batch.rew + GAMMA * qn * (1-batch.done)
    loss_q1 = torch.mean((target.detach() - qs1)**2)
    loss_q2 = torch.mean((target.detach() - qs2)**2)

    loss_c  = loss_q1 + loss_q2

    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()

    # train actor
    loss_a = torch.zeros(1)
    if timestep % D == 0:
        loss_a = torch.mean(-q1(batch.obs, pi(batch.obs)))
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()

    for po,pt in zip(q1.parameters(), tq1.parameters()):
        pt.data = pt.data * (1 - TAU) + po.data * TAU
    for po,pt in zip(q2.parameters(), tq2.parameters()):
        pt.data = pt.data * (1 - TAU) + po.data * TAU
    for po,pt in zip(pi.parameters(), tpi.parameters()):
        pt.data = pt.data * (1 - TAU) + po.data * TAU


    return {'loss_c':loss_c.item(), 'loss_a':loss_a.item()}

#==============================================================================
# RUN
env.reset()
print('Start Training...')
for ep in range(EPISODES):

    if METHOD == "prior":
        if ep % EVAL_FREQ == 0 and ep > EVAL_DELAY:
            evaluate_policy(eval_episodes=10, episode_num=ep)
    else:
        rewards, length = episode()
        writer.add_scalar('{}/rewards_training'.format(ENV), rewards, ep)
        writer.add_scalar('{}/episode_length'.format(ENV), length, ep)
        
        if ep % EVAL_FREQ == 0 and ep > EVAL_DELAY:
            evaluate_policy(eval_episodes=10, episode_num=ep)

