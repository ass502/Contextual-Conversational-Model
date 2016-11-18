from utils import *

class seq2seq_model(object):

	def __init__(self, vocab_size, batch_size, learning_rate, learning_rate_decay_factor):

		'''
		To begin: 
		- not using buckets for training
		- not using sampled softmax
		'''

		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0,trainable=False)

