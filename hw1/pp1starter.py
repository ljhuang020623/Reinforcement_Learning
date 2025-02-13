import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# B659 RL 20225 - gymnasium setup for assignment 1
# PLEASE DO NOT CHANGE ANY OF THE SETTINGS IN THIS FILE
# For your code
# please use the prepFrozen() interface for your access to the environment

np.random.seed(1)   # make sure to get repeatable results

tolerance= 0.001   # Use this for VI and iterative evaluation
discount= 1.0 -1E-3 # Use this for all algorithms and evaluation

# Note: the default 8x8 map in frozen lake is too easy
# please use the setting in this file to get the desired map

def prepFrozen(render_mode=None):
    # For reference: here are the potential arguments for generate_map
    # size: int = 8, p: float = 0.8, seed: Optional[int] = None 
    env = gym.make('FrozenLake-v1',desc=generate_random_map(seed=2),render_mode=render_mode)
    dname="Frozen8"
    env._max_episode_steps = 1000
    P=env.unwrapped.P
    nA=env.action_space.n 
    nS=env.observation_space.n 
    # Note: this is needed in order to get identical results in repeated runs
    _a,_b = env.reset(seed=1) 
    return(env,P,nS,nA,dname)

# You can use this smaller environment for debugging
# but your submission must use prepFrozen() above
def prepFrozenSmall():
    env = gym.make('FrozenLake-v1',map_name='4x4')
    dname="Frozen4"
    env._max_episode_steps = 250
    P=env.unwrapped.P
    nA=env.action_space.n 
    nS=env.observation_space.n 
    _a,_b = env.reset(seed=1) 
    return(env,P,nS,nA,dname)

# Now we can use the prep functions
# You need P for the Planning algorithms
# You need env for the evaluation, i.e., for env.reset() env.step() etc


# Demo of how to peek into and use P
# You can run this and inspect the printout to understand the structure
# of entries in P
def Pdemo():
    env,P,nS,nA,dname = prepFrozen()
    for i in range(3):
        s=np.random.randint(nS)
        a=np.random.randint(nA)
        print("at state ",s," using action ",a)
        print("Transition to: (prob,state,reward,terminated)",P[s][a])

        

# Demo of using the environment (render makes it slower)        
def rundemo():
    env,P,nS,nA,dname = prepFrozen(render_mode="human")
    observation, info = env.reset()
    terminated=False
    truncated=False
    for _i in range(100):
        action = env.action_space.sample()
        print("step: ",_i," state ",observation," action ",action)
        observation, reward, terminated, truncated, info = env.step(action)

        if (terminated or truncated):
            print("round done")
            observation, info = env.reset()
    env.close()
#Pdemo()
#rundemo()
