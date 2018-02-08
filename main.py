from agent import Agent
from net import Network
import gym
from time import time
import numpy as np
import random
#from threading import Thread

def timeit(fn):
	def wrapper(*args, **kwargs):
		start=time()
		fn(*args, **kwargs)
		print fn.__name__, "took", time()-start, "seconds."
	return wrapper

@timeit
def generate_sample_data(env, render=False, verbose=False, n_episodes=10):
	data=[]
	for ep in range(n_episodes):
		init_state=env.reset()
		prev_state=init_state
		for t in range(100):
			action=env.action_space.sample()
			new_state, reward, done, info= env.step(action)
			if done:
				reward=-10 ## modifying reward function
			if render:
				env.render()
			#curr_sample=[prev_state, action, reward, done, new_state]
			curr_sample=[]
			curr_sample.extend(prev_state)
			curr_sample.append(action)
			curr_sample.append(reward)
			curr_sample.append(done)
			curr_sample.extend(new_state)
			data.append(curr_sample)
			prev_state=new_state
			if done:
				if verbose:
					print "Episode:", ep, "score:", t
				break
	if verbose:
		print(len(data))
	data=np.array(data)
	print data.shape
	np.savetxt("data/data.csv", data, fmt='%5s', delimiter=",")

data_accumulator=[]
def write_data(data_point, thresh=500):
	global data_accumulator
	curr_sample=[]
	curr_sample.extend(data_point[0])
	curr_sample.append(data_point[1])
	curr_sample.append(data_point[2])
	curr_sample.append(data_point[3])
	curr_sample.extend(data_point[4])
	#data.append(curr_sample)
	data_accumulator.append(curr_sample)
	#print 'data acc', len(data_accumulator)
	if len(data_accumulator)==thresh:
		data_accumulator=np.array(data_accumulator)
		f=open('data/data.csv', 'a+')
		np.savetxt(f, data_accumulator, fmt='%5s', delimiter=",")
		data_accumulator=[]
		f.close()

def main_train():
	env=gym.make('CartPole-v0')
	net=Network(load_model='neural_net/net_2-14103.meta', ckpt_location='neural_net')
	a=Agent(env, 'neural_net/net_2-14103.meta', 'neural_net')

	NO_EPISODES=100
	TIMESTEPS=100
	EXPLORATION_PROB=0.2
	DISCOUNT_FACTOR=0.9 #implement
	TRAIN_EVERY_N=5
	RENDER_EVERY_N=5
	VERBOSE=True
	MODIFIED_REWARD=True
	PENALTY=-10
	WRITE_EVERY_N=50
	NO_EPOCHS=2
	BATCH_SIZE=128	
	
	for ep in range(NO_EPISODES):
		prev_state=env.reset()
		for t in range(TIMESTEPS):
			if random.uniform(0, 1)>EXPLORATION_PROB:
				action=a.get_action(prev_state)
			else:
				action=a.get_random_action()
			new_state, reward, done, info = env.step(action)
			if ep%RENDER_EVERY_N==0:
				env.render()
			if done and MODIFIED_REWARD:
				reward=PENALTY
			data_point=[prev_state, action, reward, done, new_state]
			write_data(data_point, WRITE_EVERY_N)
			prev_state=new_state
			if done:
				if VERBOSE:
					print "episode:", ep, "score:", t
				break
		#if ep%TRAIN_EVERY_N==0:
		#	net.train()

if __name__=='__main__':
	main_train()