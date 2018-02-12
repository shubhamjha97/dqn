import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
INPUT_SIZE=4
OUT_SIZE=1

def add_layer(in_layer, in_size, out_size, name):
	with tf.name_scope(name):
		W=tf.Variable(tf.random_normal(dtype=tf.float32, shape=[in_size, out_size], stddev=0.1), name='W')
		b=tf.Variable(tf.zeros(dtype=tf.float32, shape=[out_size]), name='b')
		out=tf.nn.bias_add(tf.matmul(in_layer, W), b)
	return out

class Network:

	def __init__(self, in_size=None, out_size=None, load_model=False, ckpt_location='none', save_dest='none', data_location='data/data.csv'):
		
		if not load_model:
			self.save_dest=save_dest
			self.X=tf.placeholder(tf.float32, shape=[None, in_size], name='input')
			self.y_=tf.placeholder(tf.float32, shape=[None], name='labels')
			self.action=tf.placeholder(tf.int32, shape=[None], name='actions')
			self.action_one_hot=tf.cast(tf.one_hot(self.action, 2, on_value=1, off_value=0, axis=1), tf.float32)
			self.lr=tf.placeholder(tf.float32, shape=[])

			tf.add_to_collection('input', self.X)
			tf.add_to_collection('target', self.y_)
			tf.add_to_collection('action', self.action)
			tf.add_to_collection('lr', self.lr)

			self.h1=tf.nn.sigmoid(add_layer(self.X, 4, 256, name='h1'))
			self.h2=add_layer(self.h1, 256, 512, name='h2')
			self.y=add_layer(self.h2, 512, 2, 'h3')

			self.y_masked=tf.reduce_sum(tf.multiply(self.y, self.action_one_hot), reduction_indices=[1])######################################
			self.y_max=tf.reduce_max(self.y, reduction_indices=[1], name='y_max')

			tf.add_to_collection('y', self.y)
			tf.add_to_collection('y_max', self.y_max)

			self.cost=tf.losses.mean_squared_error(labels=self.y_, predictions=self.y_masked)
			self.train_step=tf.train.AdamOptimizer(self.lr).minimize(self.cost)
			tf.add_to_collection('cost', self.cost)
			tf.add_to_collection('train_step', self.train_step)

			self.saver=tf.train.Saver(max_to_keep=5)
			#self.saver=tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=1)
			self.global_step=0

			self.sess=tf.Session()
		
			self.sess.run(tf.global_variables_initializer())
			self.saver.save(self.sess, save_dest, write_meta_graph=True)
			self.sess.run(tf.global_variables_initializer())

			
		else:
			self.save_dest=save_dest
			self.ckpt_location=ckpt_location
			self.global_step=0

			self.sess=tf.Session()    
			self.saver = tf.train.import_meta_graph(load_model)
			
			self.X=tf.get_collection('input')[0]
			self.y_=tf.get_collection('target')[0]
			self.lr=tf.get_collection('lr')[0]
			self.cost=tf.get_collection('cost')[0]
			self.train_step=tf.get_collection('train_step')[0]
			self.y=tf.get_collection('y')[0]
			self.y_max=tf.get_collection('y_max')[0]
			self.action=tf.get_collection('action')[0]
			self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_location))

	def get_batch(self, batch_size):
		my_data = np.genfromtxt('data/data.csv', delimiter=',')
		np.random.shuffle(my_data)

		state_indices=range(4)
		action_index=4
		reward_index=5
		done_index=6
		next_state_indices=range(7, 11)

		GAMMA=0.9

		for batch in range(0, my_data.shape[0], batch_size):
			state_vec = my_data[batch:batch+batch_size, state_indices]
			action_vec = my_data[batch:batch+batch_size, action_index]
			reward_vec = my_data[batch:batch+batch_size, reward_index]
			done_vec = my_data[batch:batch+batch_size, done_index]
			next_state_vec = my_data[batch:batch+batch_size, next_state_indices]

			X = state_vec
			X = X.reshape([X.shape[0], 4])

			next_X=next_state_vec
			next_X = next_X.reshape([X.shape[0], 4])

			next_Q=self.sess.run(self.y_max, feed_dict={self.X:next_X})

			y_ = reward_vec + GAMMA*np.multiply((1- done_vec), next_Q)

			yield X, y_, action_vec

	def train(self, n_epochs=10, batch_size=128, verbose=True, print_every_n=10, save_every_n=10, lr=0.01):
		for epoch in range(n_epochs):
			step=0
			for X_, labels, action_vec in self.get_batch(batch_size):
				step+=1
				self.global_step+=1
				cost_, _= self.sess.run([self.cost, self.train_step], 
					feed_dict={self.X:X_, self.y_:labels, self.action:action_vec, self.lr:lr})
				if verbose and step%print_every_n==0:
					print('epoch:', epoch, 'step:', step, 'cost', cost_) 
				if step%save_every_n==0:
					#print self.save_dest
					self.saver.save(self.sess, self.save_dest, write_meta_graph=False)

	def infer(test_data):
		logits=self.sess.run(self.y, feed_dict={self.X:np.train_data})
		return np.argmax(logits, axis=1)


if __name__=='__main__':
	#Create new net
	#net=Network(4, 1, load_model=False, save_dest='neural_net/net_2')

	#Load existing net
	net=Network(load_model='neural_net/net_2.meta', ckpt_location='neural_net', save_dest='neural_net/net_2')

	net.train(n_epochs=300, lr=0.0001)