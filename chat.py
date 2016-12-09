'''
At runtime, this script should accept checkpoint directories to figure out what model to load.

This script should accept command-line chats from the user. 
Then it will convert the user chats into one-hot encodings per the vocabulary. 
Then it will feed these chats into the encoder and receive one-hot encodings for the response.
It will print the response to the command line.
'''

import tensorflow as tf
import numpy as np
import train
import utils
import pickle
import sys
import re

tf.app.flags.DEFINE_boolean('interactive_chat', True, 'Talk to a user!')
tf.app.flags.DEFINE_string('simulate_file', '', 'File to read in for simulation')
tf.app.flags.DEFINE_string('checkpoint_dir', './models/small_model/', 'Checkpoint directory.')
tf.app.flags.DEFINE_string('vocab_dir', './data/data_idx_files/small_model_10000/', 'Checkpoint directory.')
tf.app.flags.DEFINE_boolean('argmax_decoder', False, 'How to decode')
tf.app.flags.DEFINE_integer('edit_threshold', None, 'Threshold for edit distance - None does not implement feature')

FLAGS = tf.app.flags.FLAGS


class Chat_Session(object):


	def __init__(self, sess, vocabulary_file_path, rev_vocabulary_file_path):

		#keep track of tokens - not indexes - for legibility
		self.query_log = []
		self.response_log = []

		self.vocabulary = pickle.load(open(FLAGS.vocab_dir+'vocab.p', 'rb'))
		self.reverse_vocabulary = pickle.load(open(FLAGS.vocab_dir+'rev_vocab.p', 'rb'))

		#save the session
		self.sess = sess

		#load model and set the batch value to 1
		self.model = train.create_model(self.sess,forward_only=True, checkpoint_dir=FLAGS.checkpoint_dir)
		self.model.batch_size = 1

		#get the buckets from train
		self.buckets = train._buckets


	def respond(self, idx_query):

		#check which bucket the query belongs to
		bucket_id = len(self.buckets) - 1
         	for i, bucket in enumerate(self.buckets):
				if bucket[0] >= len(idx_query):
					bucket_id = i
					break

		#This is the method from the tutorial
		if FLAGS.argmax_decoder:

			# Get a 1-element batch to feed the sentence to the model.
			encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
			  	{bucket_id: [(idx_query, [])]}, bucket_id)

			# Get output logits for the sentence. Shape of output_logits is decoder_size x vocabulary_size
			_, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
			                               target_weights, bucket_id, True)

			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			# Now outputs is list of length = decoder_size
			idx_response = [int(np.argmax(logit, axis=1)) for logit in output_logits]


		#This is the method where we condition on previous inputs
		else:

			idx_response = []
			for i in range(self.buckets[bucket_id][1]):

				# Get a 1-element batch to feed the sentence to the model.
				encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
				  	{bucket_id: [(idx_query, idx_response)]}, bucket_id)

				# Get output logits for the sentence. Shape of output_logits is decoder_size x vocabulary_size
				_, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
				                               target_weights, bucket_id, True)

				#idx_response.append( int(np.argmax(output_logits[i],axis=1)) )
				idx_response.append( utils.weighted_draw(output_logits[i][0]) )

		# If there is an EOS symbol in outputs, cut them at that point.
		if utils.EOS_ID in idx_response:
			idx_response = idx_response[:idx_response.index(utils.EOS_ID)]

		#convert the indexes to english
		response = ' '.join(self.reverse_vocabulary[i] for i in idx_response)

		return response


	def read_query(self):

		return raw_input('You: ')


	def chat(self):

		print ('HI! Let\'s chat!\n')

		while True:

			#read user query and transform it to match corpus format
			query = utils.format( self.read_query() )
			
			#transform query to idx
			idx_query = utils.sentence_to_idx(query, self.vocabulary, edit_token_threshold=FLAGS.edit_threshold, rev_vocabulary=self.reverse_vocabulary)

			#compute response to user query
			response = self.respond(idx_query)

			#print response
			print ('\nBOT: ' + response + '\n')

			#log user query and response
			self.query_log.append(query)
			self.response_log.append(response)
			

	def simulate(self, file_path):


		with open(file_path, 'rb') as simulate_file:

			#read queries and transform to idx
			queries = simulate_file.read().splitlines()
			idx_queries = [utils.sentence_to_idx(sentence, self.vocabulary, edit_token_threshold=FLAGS.edit_threshold, rev_vocabulary=self.reverse_vocabulary) for sentence in queries]

			for idx_query in idx_queries:

				#pass in query to model and decode response
				idx_response = self.respond(idx_query)

				#transform response to tokens
				response = ' '.join([self.reverse_vocabulary[i] for i in idx_response])

				#record query and response
				self.query_log.append(queries[idx])
				self.response_log.append(response)


	def save(self):
		
		filepath = ''.join(str(i) for i in np.random.randint(9, size=10))
		
		with open(filepath, 'wb') as log_file:
			
			#query log and response log should have the same length
			for idx, query in enumerate(self.query_log):

				#chats are in alternating USER QUERY - BOT RESPONSE order
				log_file.write('USER: ' + query + '\n')
				log_file.write('BOT: ' + self.response_log[idx] + '\n')

		print ('\nChat log saved to ' + filepath)


def main():

	with tf.Session() as sess:
	
		vocabulary = './'
		reverse_vocabulary = './'
		chat_session = Chat_Session(sess, vocabulary, reverse_vocabulary)
		
		if len(FLAGS.simulate_file):
			
			chat_session.simulate(FLAGS.simulate_file)
			chat_session.save()

		#this is the default 
		elif FLAGS.interactive_chat:

			try:
				chat_session.chat()

			except EOFError, KeyboardInterrupt:

				#Give user option to save chat logs before exiting
				save = raw_input('\nType \'save\' to save a log of your chats. Type anything else to exit.\n')

				if save.strip().lower() == 'save':
					chat_session.save()


if __name__ == '__main__':

	main()