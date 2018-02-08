import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
INPUT_SIZE=4
OUT_SIZE=1

def add_layer(in_layer, in_size, out_size, name):
	with tf.name_scope(name):
		W=tf.Variable(tf.random_normal(dtype=tf.float32, shape=[in_size, out_size]), name='W')
		b=tf.Variable(tf.random_normal(dtype=tf.float32, shape=[out_size]), name='b')
		out=tf.nn.bias_add(tf.matmul(in_layer, W), b)
	return out

class Network:

	def __init__(self, in_size=None, out_size=None, load_model=False, ckpt_location='none', save_dest='none', data_location='data/data.csv'):

		if not load_model:
			self.X=tf.placeholder(tf.float32, shape=[None, in_size], name='input')
			self.y_=tf.placeholder(tf.float32, shape=[None, out_size], name='labels')
			tf.add_to_collection('input', self.X)

			self.h1=add_layer(self.X, 4, 128, name='h1')
			self.h2=add_layer(self.h1, 128, 256, name='h2')
			self.y=tf.reduce_max(tf.sigmoid(add_layer(self.h2, 256, 2, 'h3'), name='y'),
				reduction_indices=[1])
			tf.add_to_collection('y', self.y)

			self.cost=tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
			self.train_step=tf.train.AdamOptimizer(0.01).minimize(self.cost)
			tf.add_to_collection('cost', self.cost)
			tf.add_to_collection('train_step', self.train_step)

			self.saver=tf.train.Saver(max_to_keep=5)
			self.global_step=0

			self.sess=tf.Session()
		
			self.sess.run(tf.global_variables_initializer())
			self.saver.save(self.sess, save_dest, global_step=self.global_step)
			
		else:
			self.sess=tf.Session()    
			self.saver = tf.train.import_meta_graph(load_model)
			self.saver.restore(self.sess,tf.train.latest_checkpoint(ckpt_location))

	def get_batch(self, batch_size):
		my_data = np.genfromtxt('data/data.csv', delimiter=',')
		np.random.shuffle(my_data)
		for batch in range(0, my_data.shape[0], batch_size):
			X = my_data[batch:batch+batch_size, 0:4]
			X = X.reshape([X.shape[0], 4])
			next_X=my_data[batch:batch+batch_size, 7:]
			next_X = next_X.reshape([X.shape[0], 4])
			print next_X.shape
			next_Q=self.sess.run(self.y, feed_dict={self.X:next_X})
			next_Q=next_Q.reshape(next_Q.shape[0],1)
			y_ = my_data[batch:batch+batch_size, 2] + my_data[batch:batch+batch_size, 3]*next_Q
			yield X, y_

	def train(self, n_epochs=1, batch_size=128, verbose=True, print_every_n=50):
		for epoch in range(n_epochs):
			step=0
			for X_, labels in self.get_batch(batch_size):
				step+=1;
				cost_, _= self.sess.run([self.cost, self.train_step],
					feed_dict={self.X:X_, self.y_:labels})
				if verbose and step%print_every_n==0:
					print 'epoch:', epoch, 'step:', step, 'cost', cost_ 

	def infer(test_data):
		logits=self.sess.run(self.y, feed_dict={self.X:np.train_data})
		return np.argmax(logits, axis=1)


if __name__=='__main__':
	#net=Network('neural_net/test_model-0.meta', 'neural_net')
	net=Network(4, 1, load_model=False, save_dest='neural_net/net_2')
	#net=Network(load_model='neural_net/net_2.meta', ckpt_location='neural_net')
	net.train()