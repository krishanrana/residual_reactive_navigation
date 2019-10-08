import numpy as np, cv2, math, random, time
from PointGoalNavigationEnv_v0 import *
from prior_controller import *

#==============================================================================
# ENV PARAMS

HYPERS = dict(# training params
              viz                = "--viz" in sys.argv,
              log                = "--log" in sys.argv,
              use_cuda           = "--cuda" in sys.argv,
              load               = str  (sys.argv[sys.argv.index("--load"   )+1]) if "--load"    in sys.argv else None,
              seed               = int  (sys.argv[sys.argv.index("--seed"   )+1]) if "--seed"    in sys.argv else int(time.time()*1e6)%2**30,
              runtime            = float(sys.argv[sys.argv.index("--runtime")+1]) if "--runtime" in sys.argv else None,
              ckpt_interval      = 10000,
              log_interval       = 1000,
              eval_interval      = 10000,

              # env params
              num_beams          = 270,
              laser_range        = 0.5,
              laser_noise        = 0.01,
              angle_min          = 3*-np.pi/4,
              angle_max          = 3*np.pi/4,
              timeout            = 300,
              velocity_max       = 1,
              omega_max          = 1,
              env_type           = 1,  

)

for k,v in HYPERS.items(): exec("{} = {!r}".format(k,v))

#==============================================================================

env = PointGoalNavigation(**HYPERS)
obs = env.reset()
prior = PotentialFieldsController()

while(1):
    dist_to_goal, angle_to_goal, robot_loc, robot_angle, laser_scan, goal_loc, obs_loc = env._get_position_data()
    action = prior.computeResultant(dist_to_goal, angle_to_goal, laser_scan)
    obs, rews, done, _ = env.step(action)
    env.render()
    if done:
      obs = env.reset()

