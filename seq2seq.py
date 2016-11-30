from __future__ import division

import random
import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import utils

class seq2seq_model(object):

	def __init__(self, vocab_size, buckets, size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, num_samples=512, forward_only=False,dtype=tf.float32):

		'''
		To begin: 
		- not using buckets for training
		- not using sampled softmax
		'''

		self.vocab_size = vocab_size
		
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0,trainable=False)

		# If we use sampled softmax, we need an output projection.
		output_projection = None
		softmax_loss_function = None

		# Sampled softmax only makes sense if we sample less than vocabulary size.
		if num_samples > 0 and num_samples < self.vocab_size:
			w_t = tf.get_variable("proj_w", [self.vocab_size, size], dtype=dtype)
			w = tf.transpose(w_t)
			b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
			output_projection = (w, b)

			def sampled_loss(inputs, labels):
				labels = tf.reshape(labels, [-1, 1])
				# We need to compute the sampled_softmax_loss using 32bit floats to avoid numerical instabilities.
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b, tf.float32)
				local_inputs = tf.cast(inputs, tf.float32)
				return tf.cast(tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels, num_samples, self.vocab_size), dtype)
		

		"""def full_loss(inputs,labels):
				local_inputs = tf.cast(inputs, tf.float32)
				return tf.cast(tf.nn.softmax_cross_entropy_with_logits(inputs, labels), dtype)"""

		softmax_loss_function = sampled_loss
		#softmax_loss_function = full_loss


		single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
		cell = single_cell
		if num_layers>1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

		# The seq2seq function: we use embedding for the input
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			#Choose which type of seq2seq model to use
			"""return tf.nn.seq2seq.embedding_attention_seq2seq(
				encoder_inputs,
				decoder_inputs,
				cell,
				num_encoder_symbols = vocab_size,
				num_decoder_symbols = vocab_size,
				embedding_size = size,
				output_projection = None,
				feed_previous = do_decode,
				dtype = dtype)"""
			return tf.nn.seq2seq.embedding_rnn_seq2seq(
				encoder_inputs,
				decoder_inputs,
				cell,
				num_encoder_symbols=vocab_size,
				num_decoder_symbols=vocab_size,
				embedding_size=size,
				output_projection=output_projection,
				feed_previous=do_decode,
				dtype=dtype)

		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []

		for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
		for i in xrange(buckets[-1][1] + 1):
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                            	name="weight{0}".format(i)))

		# Our targets are decoder inputs shifted by one.
		targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

		# Training outputs and losses.
		if forward_only:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
			self.encoder_inputs, self.decoder_inputs, targets,
			self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
			softmax_loss_function=softmax_loss_function)

			# If we use output projection, we need to project outputs for decoding.
			if output_projection is not None:
				for b in xrange(len(buckets)):
					self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]
		else:
			self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
			self.encoder_inputs, self.decoder_inputs, targets,
			self.target_weights, buckets,
			lambda x, y: seq2seq_f(x, y, False),
			softmax_loss_function=softmax_loss_function)

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []

			#Choose which optimizer to use
			opt = tf.train.AdamOptimizer(self.learning_rate)
			#opt = tf.train.GradientDescentOptimizer(self.learning_rate)

			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
				self.gradient_norms.append(norm)
				self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

		self.saver = tf.train.Saver(tf.all_variables())

	def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only, input_batch_size=None):
		# Check if the sizes match.
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
							" %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
							" %d != %d." % (len(decoder_inputs), decoder_size))
		if len(target_weights) != decoder_size:
			raise ValueError("Weights length must be equal to the one in bucket,"
							" %d != %d." % (len(target_weights), decoder_size))

		#either use provided batch size (to eval entire dev/test set) or use model default for training
		if input_batch_size is None:
			batch_size = self.batch_size
		else:
			batch_size = input_batch_size

		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		last_target = self.decoder_inputs[decoder_size].name
		input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
							self.gradient_norms[bucket_id],  # Gradient norm.
							self.losses[bucket_id]]  # Loss for this batch.
		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


	def get_batch(self, data, bucket_id, input_batch_size=None):
		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, decoder_inputs = [], []

		#either use provided batch size (to eval entire dev/test set) or use model default for training
		if input_batch_size is None:
			batch_size = self.batch_size

			# Get a random batch of encoder and decoder inputs from data,
			# pad them if needed, reverse encoder inputs and add GO to decoder.
			for _ in xrange(self.batch_size):
				encoder_input, decoder_input = random.choice(data[bucket_id])

				# Encoder inputs are padded, but not reversed in the conversational setting
				encoder_pad = [utils.PAD_ID] * (encoder_size - len(encoder_input))
				encoder_inputs.append(list(encoder_input + encoder_pad))

				# Decoder inputs get an extra "GO" symbol, and are padded then.
				decoder_pad_size = decoder_size - len(decoder_input) - 1
				decoder_inputs.append([utils.GO_ID] + decoder_input + [utils.PAD_ID] * decoder_pad_size)
		else:
			batch_size = input_batch_size

			#instead of getting a random batch, select the appropriate number at once from the data
			# pad them if needed, reverse encoder inputs and add GO to decoder.
			for i in xrange(batch_size):
				encoder_input, decoder_input = data[bucket_id][i]

				# Encoder inputs are padded, but not reversed in the conversational setting
				encoder_pad = [utils.PAD_ID] * (encoder_size - len(encoder_input))
				encoder_inputs.append(list(encoder_input + encoder_pad))

				# Decoder inputs get an extra "GO" symbol, and are padded then.
				decoder_pad_size = decoder_size - len(decoder_input) - 1
				decoder_inputs.append([utils.GO_ID] + decoder_input + [utils.PAD_ID] * decoder_pad_size)

		# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

		# Batch encoder inputs are just re-indexed encoder_inputs.
		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32))

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for length_idx in xrange(decoder_size):
			batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32))

			# Create target_weights to be 0 for targets that are padding.
			batch_weight = np.ones(batch_size, dtype=np.float32)
			for batch_idx in xrange(batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				if length_idx == decoder_size - 1 or target == utils.PAD_ID:
					batch_weight[batch_idx] = 0.0
			batch_weights.append(batch_weight)
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights

