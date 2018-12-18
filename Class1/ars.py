# AI 2018

# Importing Libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):
        self.nb_steps = 3
        # Max time we allow for the AI to walk on the field (free to try an error)
        self.episode_length = 1000
        self.learning_rate = 0.02
        # The more directions, the higher chance of reward, but with a longer time to train
        self.nb_directions = 16
        # For saving the best direction
        self.nb_best_directions = 16
        # Best directions should be lower than the directions
        assert self.nb_best_directions <= self.nb_directions
        # Standard deviation signal
        self.noise = 0.03
        # To fix the parameter of the environment (to get the same result everytime (for debuggin purpose))
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'
        
# Normalizing the sates

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
        ## print('Init Normalizer.n...', self.n)
        
    def observe(self, x):
        ## print('observe x: ', x)
        self.n += 1.
        ## print('self.n: ', self.n)
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        ## print('mean: ', self.mean)
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        ## print('self.mean_diff: ', self.mean_diff)
        ## print('self.var: ', self.var)
        
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    
# Building the AI
        
class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
        ## print('Init policy...')
        ## print('self.theta: ', self.theta)
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        # print('policy.sample_deltas(): ', [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)])
        # First we put randn for array of size self.theta (Look at init),
        # Then we apply the same thing for 16 times (because our  positive direction is 16)
        # Then another 16 for negative direction, but we will use the same value as positive direction
        # So total is 32 (just 16 actually (same value as positive direction)) of array with theta.shape
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        
# Exploring the policy on one specific direction and over one episode
# Because in one episode there will be many actions (ie. the motor, the gyrometer, other gyro, etc...)
# For each actions have their rewards, and we will get the acumulate rewards
        
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    ## print('state RESET: ', state)
    done = False
    num_plays = 0.
    sum_rewards = 0
    # Calculate the accumulate reward on the full episode
    while not done and num_plays < hp.episode_length:
        ## print('state B4: ', state)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        ## print('action: ', action)
        # This pybullet library will return the next state, reward, is the episode is done, and 1 more...
        state, reward, done, _ = env.step(action)
        ## print('state AFTER: ', state)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training AI
    
def train(env, policy, normalizer, hp):
    
    print('hp.nb_steps: ', hp.nb_steps)
    
    for step in range(hp.nb_steps):
        
        ## print('step in range(hp.nb_steps): ', step)
        
        # Initializing the pertubation deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        ## print('deltas: ', deltas)
        positive_rewards = [0] * hp.nb_directions
        ## print('positive_rewards: ', positive_rewards)
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            ## print('deltas[', k,']: ', deltas[k])
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
            ## print('positive_rewards[', k,']: ', positive_rewards[k])
        ## print('positive_rewards: ', positive_rewards)
            
        # Getting the negative rewards in the negative/positive directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
            
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating the policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Rewards: ', reward_evaluation)
        
# Running the main code
print('START')
        
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
print('hp.seed: ', hp.seed)
print('np.random.seed: ', np.random.rand(1))
env = gym.make(hp.env_name)
print('First env: ', env)
env = wrappers.Monitor(env, monitor_dir, force = True)
print('Second env: ', env)
nb_inputs = env.observation_space.shape[0]
print('nb_inputs: ', nb_inputs)
nb_outputs = env.action_space.shape[0]
print('nb_outputs: ', nb_outputs)
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
print('EXIT')