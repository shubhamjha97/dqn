import gym
import numpy as np
import tensorflow as tf
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def timeit(fn):
	def wrapper(*args, **kwargs):
		start=time()
		result=fn(*args, **kwargs)
		print fn.__name__, "took", time()-start, "seconds."
		return result
	return wrapper

class Agent:
	def __init__(self, env, model_path, ckpt_path):
		self.env=env
		self.sess=tf.Session()    
		self.saver = tf.train.import_meta_graph(model_path)
		self.saver.restore(self.sess,tf.train.latest_checkpoint(ckpt_path))
		self.y = tf.get_collection('y')[0]
		self.X = tf.get_collection('input')[0]
		#print all variables in graph
		#print [n.name for n in tf.get_default_graph().as_graph_def().node]
	
	#@timeit
	def get_action(self, curr_state):
		logits=self.sess.run(self.y, feed_dict={self.X:np.array(curr_state).reshape(1, 4)})
		return np.argmax(logits)

	def get_random_action(self):
		return self.env.action_space.sample()

if __name__=='__main__':
	import gym
	env=gym.make('CartPole-v0')
	env.reset()
	a=Agent(env, 'neural_net/net_2-0.meta', 'neural_net')
	#action=env.action_space.sample()
	for _ in range(1):
		for __ in range(100):
			new_state, reward, done, info= env.step(a.get_random_action())
			print a.get_action(new_state)
			if done:
				break