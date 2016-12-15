'''
At runtime, this script should accept checkpoint directories to figure out what model to load.

This script should accept command-line chats from the user. 
Then it will convert the user chats into one-hot encodings per the vocabulary. 
Then it will feed these chats into the encoder and receive one-hot encodings for the response.
It will print the response to the command line.
'''

import modified_utils
import tensorflow as tf
import numpy as np
import train
import utils
import cbow_utils
import fasttext
import pickle
import sys
import re

tf.app.flags.DEFINE_boolean('interactive_chat', True, 'Talk to a user!')
tf.app.flags.DEFINE_string('simulate_file', '', 'File to read in for simulation')
tf.app.flags.DEFINE_string('checkpoint_dir', './models/debug_dir_UNK/', 'Checkpoint directory.')
tf.app.flags.DEFINE_string('vocab_dir', './data/data_idx_files/small_model_100000_unks/', 'Checkpoint directory.')
tf.app.flags.DEFINE_boolean('argmax_decoder', True, 'How to decode')
tf.app.flags.DEFINE_integer('sample_k', None, 'Top k logits to sample for in sampled decoding - None samples over all words')
tf.app.flags.DEFINE_integer('edit_threshold', None, 'Threshold for edit distance - None does not implement feature')
tf.app.flags.DEFINE_boolean('special_unks', False, 'Use special unks for decoding and training')
tf.app.flags.DEFINE_string('cbow_model', '', 'Use CBOW predictions for decoding and training')
tf.app.flags.DEFINE_float('replace_prob', 1.0, 'Replace prob for CBOW substitutions')

FLAGS = tf.app.flags.FLAGS

class Chat_Session_UNKS(object):

	'''CHAT CLASS with UNKS tweaks: has the special unks functionality. options for greedy and argmax decoder, 
	and loading a pre-trained CBOW model to guess unknown words'''

	def __init__(self, sess):

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

		#open the CBOW model
		if len(FLAGS.cbow_model) > 0:
			self.cbow_model = fasttext.load_model(FLAGS.cbow_model, label_prefix='__LABEL__')
		else:
			self.cbow_model = None

		#store the special unk chars in the model
		self.special_unk_idxes = [modified_utils.CAPS_UNK_ID_1, modified_utils.CAPS_UNK_ID_2, modified_utils.CAPS_UNK_ID_3]


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

				idx_response.append( utils.weighted_draw(output_logits[i][0],FLAGS.sample_k) )

		# If there is an EOS symbol in outputs, cut them at that point.
		if utils.EOS_ID in idx_response:
			idx_response = idx_response[:idx_response.index(utils.EOS_ID)]

		#convert the indexes to english
		#response = ' '.join(self.reverse_vocabulary[i] for i in idx_response)

		return idx_response


	def read_query(self):

		return raw_input('You: ')


	def chat(self):

		print ('HI! Let\'s chat!\n')

		while True:

			#read user query and transform it to match corpus format
			query = modified_utils.format( self.read_query() )
			
			#transform query to idx
			idx_query, special_unk_assignments, cbow_guesses = modified_utils.sentence_to_idx(query, self.vocabulary, cbow=self.cbow_model, replace_prob=FLAGS.replace_prob)
			
			#compute response to user query
			idx_response = self.respond(idx_query)

			#convert the response to actual tokens...
			response = [-1]*len(idx_response)

			#reverse the special_unk_assignments dict: now you have the unk_idx as key and the special token as value
			reversed_special_unk_assignments = {val: key for key, val in special_unk_assignments.iteritems()}

			#see if any of the special_unk_assignment tokens are in the idx response
			for i, vocab_idx in enumerate(idx_response):
				try:
					response[i] = reversed_special_unk_assignments[vocab_idx]
				except KeyError:
					pass

			#do normal vocabulary mapping for everything else
			for i, vocab_idx in enumerate(idx_response):
				try:
					response[i] = self.reverse_vocabulary[vocab_idx]
				except KeyError:
					response[i] = self.reverse_vocabulary['_UNK_']

			#reverse the cbow guesses dict: now you have the CBOW replacement as key and the original token as value
			reversed_cbow_guesses = {val: key for key, val in cbow_guesses.iteritems()}

			#see if any cbow guesses ended up in the response
			for i, token in enumerate(response):
				if token in reversed_cbow_guesses:
					response[i] = reversed_cbow_guesses[token]

			#print response
			print ('\nBOT: ' + ' '.join(i for i in response) + '\n')

			#log user query and response
			self.query_log.append(query)
			self.response_log.append(response)
			

	def simulate(self, file_path):

		with open(file_path, 'rb') as simulate_file:

			#read queries and transform to idx
			queries = simulate_file.read().splitlines()

			for query in queries:

				#read user query and transform it to match corpus format
				query = utils.format( query() )
				
				#transform query to idx
				idx_query, special_unk_assignments, cbow_guesses = modified_utils.sentence_to_idx(query, self.vocabulary, cbow_model=self.cbow_model, replace_prob=FLAGS.replace_prob)

				#compute response to user query
				idx_response = self.respond(idx_query)

				#convert the response to actual tokens...
				response = [-1]*len(idx_response)

				#reverse the special_unk_assignments dict: now you have the unk_idx as key and the special token as value
				reversed_special_unk_assignments = {val: key for key, val in special_unk_assignments.iteritems()}

				#see if any of the special_unk_assignment tokens are in the idx response
				for i, idx in enumerate(idx_reponse):
					try:
						response[i] = reversed_special_unk_assignments[idx]
					except KeyError:
						pass

				#do normal vocabulary mapping for everything else
				for i, idx in enumerate(idx_response):
					try:
						response[i] = self.reverse_vocabulary[i]
					except KeyError:
						response[i] = self.reverse_vocabulary['_UNK_']

				#reverse the cbow guesses dict: now you have the CBOW replacement as key and the original token as value
				reversed_cbow_guesses = {val: key for key, val in cbow_guesses.iteritems()}

				#see if any cbow guesses ended up in the response
				for i, token in enumerate(response):
					if token in reversed_cbow_guesses:
						response[i] = reversed_cbow_guesses[token]

				#print response
				print ('\nBOT: ' + response + '\n')

				#log user query and response
				self.query_log.append(query)
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


'''
================================================================================================================================
	ORIGINAL CHAT CLASS BELOW
================================================================================================================================
'''
class Chat_Session(object):

	'''ORIGINAL CHAT CLASS: this has the default behavior, with options for greedy vs sampled decoding, and edit distance''' 

	def __init__(self, sess):

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

				idx_response.append( utils.weighted_draw(output_logits[i][0],FLAGS.sample_k) )

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

		#load up the appropriate chat session object
		if FLAGS.special_unks:
			chat_session = Chat_Session_UNKS(sess)
		else:
			chat_session = Chat_Session(sess)
		
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