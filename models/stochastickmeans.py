import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def tf_to_unitvec(x):
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), 1, keep_dims=True))
    return x / norm 

def tf_cosine_matrix(a,b):
    # a: matrix, b: vector
    # out: cosine(matrix_i,b)
    c=tf.sqrt(tf.reduce_sum(tf.multiply(a,a),axis=1)) 
    d=tf.sqrt(tf.reduce_sum(tf.multiply(b,b),axis=1)) 
    e=tf.reduce_sum(tf.multiply(a,b),axis=1)
    f=tf.multiply(c,d)
    r=tf.div(e,f)

    return r

class StochasticKmeans(object):
    def __init__(self,n, distance = 'l2', case = 'train', square=False, reduce_n = False, W0=None): 
        print('Stochastic k-means')
        self.size = 300
        self.n = n
        self.distance = distance
        self.case = case
        self.square_norm = square
        self.reduce_n = reduce_n
        
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.X = tf.placeholder(tf.float32, shape=(None, self.size), name= 'X') 
 
        if case =='train':
            self.W = tf.Variable(tf.cast(W0, tf.float32), name='{}/W_topic'.format(case))
        elif case == 'test':
            self.W = tf.placeholder(tf.float32, shape=(self.n, 300), name= '{}/W_topic'.format(case))
        else:
            raise

        self.y_true = tf.placeholder(tf.int32, shape=(None), name= 'y_true') 

        hilf, self.loss_1, self.loss_2 = [],[],[]    

        self.W = tf_to_unitvec(self.W) 

        l2 = tf.stack([tf.square(tf.norm(self.X-self.W[i,:], axis=1)) for i in range(self.n)])
        inds, mins = tf.argmin(l2, axis=0), tf.reduce_min(l2, axis=0)

        self.loss = tf.reduce_mean(mins)

        self.loss_sep = (tf.cast(1.0, tf.float32), tf.cast(1.0, tf.float32))
        
        self.acc = tf.cast(1.0, tf.float32)

         
        if self.case == 'train':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
        elif self.case == 'test':
            print('No training')
            
        self.grad = tf.gradients(self.loss, self.W)            
