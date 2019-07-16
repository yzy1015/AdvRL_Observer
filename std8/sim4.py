import sys
sys.path.insert(0, './py_torch_trpo')
from baselines.common import set_global_seeds, tf_util as U
import gym
import roboschool
import numpy as np
import random
from expert import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from gym import spaces
from new_baseline_model.mlp import MlpPolicy_new
from new_baseline_model.TRPO_agent import TRPO_agent_new
from baselines import logger
import pickle
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy, pposgd_simple
import os

plt.style.use('seaborn-white')
sns.set(context = "paper", font = "monospace", font_scale=2)
logger.configure()
U.make_session(num_cpu=16).__enter__()
#set_global_seeds(seed)

folder = 'simu_update_4'
os.mkdir(folder)
env_name = "RoboschoolInvertedPendulum-v1"
global_env = gym.make(env_name)


# dummy class, use for construction of adversary and observer environment
class dummy_adversary_env(object):
    def __init__(self):
        self.env = global_env
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.env.observation_space.shape[0],))
        self.observation_space = self.env.observation_space
        
    def action_ev(self, s):
        return action_space.sample()
    
    def reset(self):
        return self.env.reset()
    
class dummy_observer_env(object):
    def __init__(self):
        self.env = global_env
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.env.observation_space.shape[0],))
        self.observation_space = self.env.observation_space
    
    def action_ev(self, s):
        return s
    
    def reset(self):
        return self.env.reset()

env2 = dummy_adversary_env()
class pargm(object):
    def __init__(self):
        self.timesteps_per_batch = 25000 # what to train on
        self.max_kl = 0.01
        self.cg_iters = 10
        self.gamma = 0.995
        self.lam =  0.97# advantage estimation
        self.entcoeff=0.0
        self.cg_damping=0.1
        self.vf_stepsize=1e-3
        self.vf_iters =5
        self.callback=None

    def policy_func(self, name, ob_space, ac_space):
        return MlpPolicy_new(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)
    
    
parg = pargm()
adversary = TRPO_agent_new('pi1', env2, parg)

env3 = dummy_observer_env()
class pargm1(object):
    def __init__(self):
        self.timesteps_per_batch = 25000 # what to train on
        self.max_kl = 0.01
        self.cg_iters = 10
        self.gamma = 0.995
        self.lam =  0.97# advantage estimation
        self.entcoeff=0.0
        self.cg_damping=0.1
        self.vf_stepsize=1e-3
        self.vf_iters =5
        self.callback=None

    def policy_func(self, name, ob_space, ac_space):
        return MlpPolicy_new(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)
    
    
parg1 = pargm1()
observer = TRPO_agent_new('pi2', env3, parg1)

class env_wrapper(object):
    def __init__(self, en_name):
        self.env = gym.make(en_name)
        self.agent = SmallReactivePolicy(self.env.observation_space, self.env.action_space) # declare sample trained agent
        
    def reward(self, st):
        return np.abs(st[3])-0.08
    
    def step(self, obs):
        ac = self.agent.act(obs)
        obsr, r, done, temp = self.env.step(ac)
        return obsr, self.reward(obsr), done, temp
    
    def reset(self):
        return self.env.reset()
    
threshold = np.array([ 0,  0,  0,  0.00789366,  0])*8
env = env_wrapper(env_name)
score_avg = []
score_ll = []
for i in range(60000):
    score = 0
    obs = env.reset()
    done = False
    while done == False:   
        a_noise = adversary.action_st(obs)
        a_noise = np.clip(a_noise,-1,1)
        #print(a_noise*threshold)
        obs_ad = obs + a_noise*threshold       
        a = observer.action_st(obs_ad)
        obs, r, done, _ = env.step(a)
        #print(obs, r, done)
        l1 = adversary.obtain_reward(r, done)
        l2 = observer.obtain_reward(-r, done)
        score += r
        if l1:
            print('adv learn')
            
        if l2:
            print('obs learn')
    
    score_ll.append(score)
    
    if i%20 == 19:
        print(np.mean(score_avg))
        score_avg = []
        pickle_out = open(folder+"/score.pickle","wb")
        pickle.dump(score_ll, pickle_out)
    #print(score)
    
    score_avg = score_avg + [score]
    
pickle_out = open(folder+"/score.pickle","wb")
pickle.dump(score_ll, pickle_out)
