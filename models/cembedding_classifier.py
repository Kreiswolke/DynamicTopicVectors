import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def tf_to_unitvec(x):
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), 1, keep_dims=True))
    return x / norm 


class EmbeddingClassifier(object):
    def __init__(self,n, case = 'train'): 
        print('Stochastic k-means')
        self.size = 300
        self.n = n
        self.case = case

        
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.X = tf.placeholder(tf.float32, shape=(None, self.size), name= 'X') 

        self.W = tf.placeholder(tf.float32, shape=(self.n, 300), name= '{}/W_topic'.format(case))

        self.y_true = tf.placeholder(tf.int32, shape=(None), name= 'y_true') 
        self.y_true_onehot = tf.one_hot(self.y_true, self.n)

     
        self.output = tf.matmul(self.X, tf.transpose(self.W)) # [batchsize x num_topics]
        
        self.final_output = tf.contrib.layers.fully_connected(self.output, n_outputs, activation_fn=tf.nn.relu, scope='stv_supervised')

        self.hilf =  tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_output, labels=self.y_true_onehot)         
         
        self.loss = tf.reduce_mean(self.hilf) # Average over N_docxN_topics loss matrix
            
        self.acc = tf.metrics.accuracy(self.y_true, tf.argmax(self.final_output, axis=1))

        
        if self.case == 'train':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
        elif self.case == 'test':
            print('No training')
            
