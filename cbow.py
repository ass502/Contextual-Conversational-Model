'''
Train CBOW model using python wrapper for Facebook FastText implementation. 
'''

import fasttext

train_data_path = './data/cbow_data_100000/cbow_TRAIN_100002.data'
dev_data_path = './data/cbow_data_100000/cbow_DEV_100002.data'

classifier = fasttext.supervised(train_data_path, 'cbow_model', label_prefix='__label__')

train_results = classifier.test(train_data_path)
print 'train precision-1: ' + str(train_results.precision)
print 'train recall: ' + str(train_results.recall)

dev_results = classifier.test(dev_data_path)
print 'dev precision-1: ' + str(dev_results.precision)
print 'dev recall: ' + str(dev_results.recall)