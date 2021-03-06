from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import os.path
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils
import seq2seq

#patience_threshold = 1
MAX_NUM_STEPS = 50000
vocab_size = 100000
size = 1024
num_layers = 2
max_gradient_norm = 5.0
batch_size = 32

#for gradient descent, initial lr is 0.5
#learning_rate = 0.5
learning_rate_decay_factor = 1

#for Adam optimizer, initial lr is .0001
learning_rate = .0001

steps_per_checkpoint = 1000
steps_per_print = 200
use_fp16 = False

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("data_dir", "./data/data_idx_files/1st_run_50000/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./1st_baseline/", "Training directory.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5,5), (10, 10), (25, 25), (40,40)]


def read_data(input_path, output_path, max_size=None):
	"""Read data from input and output files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the input sentence.
    target_path: path to the file with token-ids for the target sentence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (input, output) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(input) < _buckets[n][0] and
      len(output) < _buckets[n][1]; source and target are lists of token-ids.
  """
	data_set = [[] for _ in _buckets]
	with tf.gfile.GFile(input_path, mode="r") as source_file:
		with tf.gfile.GFile(output_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				if counter % 100000 == 0:
					print("  reading data line %d" % counter)
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()

	return data_set


def create_model(session, forward_only, checkpoint_dir=None):
	"""Create conversational model and initialize or load parameters in session."""
	dtype = tf.float16 if use_fp16 else tf.float32
	model = seq2seq.seq2seq_model(
		vocab_size,
		_buckets,
		size,
		num_layers,
		max_gradient_norm,
		batch_size,
		learning_rate,
		learning_rate_decay_factor,
		forward_only=forward_only,
		dtype=dtype)

	if checkpoint_dir is None:
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	else:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())

	return model

def get_data_paths(data_dir, vocab_size):

	return data_dir+"train_"+str(vocab_size)+".example", data_dir+"train_"+str(vocab_size)+".label", \
			data_dir+"dev_"+str(vocab_size)+".example", data_dir+"dev_"+str(vocab_size)+".label",  \
			data_dir+"test_"+str(vocab_size)+".example", data_dir+"test_"+str(vocab_size)+".label"

def train():
	"""Train a conversational model"""

	# Prepare data.
	print("Preparing data in %s" % FLAGS.data_dir)
	
	in_train, out_train, in_dev, out_dev, in_test, out_test = get_data_paths(FLAGS.data_dir, vocab_size)

	with tf.Session() as sess:
	# Create model.
		print("Creating %d layers of %d units." % (num_layers, size))
		model = create_model(sess, False)

		# Read data into buckets and compute their sizes.
		print ("Reading development and training data.")
		dev_set = read_data(in_dev, out_dev)
		train_set = read_data(in_train, out_train)
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		if os.path.isfile(FLAGS.data_dir+'curves.txt'):
			curves_path = FLAGS.data_dir+'curves'+str(np.random.randint(10**8))+'.txt'
		else:
			curves_path = FLAGS.data_dir+'curves.txt'

		with open(curves_path,'wb') as out_f:
			out_f.write("Train and Perplexity Curves:\n\n")

		# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
		# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
		# the size if i-th training bucket, as used later.
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		#previous_losses = []

		'''
		#keep track of previous perplexity and patience for early stopping
		prev_eval_loss = 10**6
		patience_count = 0
		keep_training = True

		while keep_training:
		'''
		while current_step <= MAX_NUM_STEPS:

			# Choose a bucket according to data distribution. We pick a random number
			# in [0, 1] and use the corresponding interval in train_buckets_scale.
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
			step_time += (time.time() - start_time) / steps_per_checkpoint
			loss += step_loss / steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % steps_per_print == 0:
				
				# Print statistics for the previous epoch.
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
				print("global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
				with open(curves_path,'ab') as out_f:
					out_f.write("global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
					out_f.write('\n')
				
				# Decrease learning rate if no improvement was seen over last 3 times.
				#if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				#	sess.run(model.learning_rate_decay_op)
				#previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				if current_step % steps_per_checkpoint == 0:
					checkpoint_path = os.path.join(FLAGS.train_dir, "conversation.ckpt")
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				
				# Run evals on development set and print their perplexity.
				checkpoint_eval_loss = 0
				for bucket_id in xrange(len(_buckets)):
					if len(dev_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
						continue

					encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id, input_batch_size = None)
					
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, input_batch_size = None)
				  
					eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
					with open(curves_path,'ab') as out_f:
						out_f.write("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
						out_f.write("\n\n")

				  	checkpoint_eval_loss += eval_loss

				'''
				if checkpoint_eval_loss > prev_eval_loss:
					patience_count += 1

				prev_eval_loss = checkpoint_eval_loss

				if patience_count == patience_threshold:
					keep_training = False
					print ('Overfitting occurred at step: ' + str(current_step))
				'''

				sys.stdout.flush()


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
