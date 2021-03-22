# -*- coding: utf-8 -*-
"""
MOT-Price Approximation with Neural Networks
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import layers, Model
from tensorflow.keras.constraints import non_neg
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def mot_nn_bidask(strikes1,
                  prices1_bid,
                  prices1_ask,
                  strikes2,
                  prices2_bid,
                  prices2_ask,
                  s0=100,
                  lower_bound=True,
                  from_discrete= 0,
                  to_discrete = 2,
                  kappa = 0,
                  batch_size = 2**8,
                  nr_iterations = 2000,
                  gamma = 100):
    
    # Setting all variables to be of float32-type
    tf.keras.backend.set_floatx('float32')
    strikes1 = strikes1 / s0
    prices1_bid = prices1_bid / s0
    prices1_ask = prices1_ask / s0
    strikes2 = strikes2 / s0
    prices2_bid = prices2_bid / s0
    prices2_ask = prices2_ask / s0
    kappa = kappa / s0
    
    def payoff(x,y):
        return tf.abs(x-y)
    
    def generate_samples(batch_size):
        while True:
            x = tf.random.uniform(shape = [batch_size,1])*(to_discrete-from_discrete)+from_discrete
            yield x
        
    # Define the Loss Function
    def loss(bias,Delta_0,model_u1_bid,model_u1_ask,model_u2_bid,model_u2_ask,model_Del,x,y):
        u_1 = model_u1_bid(x)
        u_1 = u_1 + model_u1_ask(x)
        u_2 = model_u2_bid(y)
        u_2 = u_2 + model_u2_ask(y)
        u_1 = tf.reshape(tf.repeat(u_1,batch_size),(batch_size,batch_size))
        u_2 = tf.transpose(tf.reshape(tf.repeat(u_2,batch_size),(batch_size,batch_size)))
        Delta_output = model_Del(x)
        Delta = tf.reshape(tf.repeat(Delta_output,batch_size),(batch_size,batch_size))
        Delta = tf.math.multiply(Delta,tf.transpose(tf.reshape(tf.repeat(y,batch_size),(batch_size,batch_size)))-tf.reshape(tf.repeat(x,batch_size),(batch_size,batch_size)))
        payoff_vec = payoff(x*s0,tf.transpose(y*s0))*(1-2*lower_bound)/s0
        bias = tf.reshape(tf.repeat(bias,batch_size**2),(batch_size,batch_size))
        transaction_costs = kappa*(tf.reshape(tf.repeat(tf.abs(Delta_0),batch_size**2),(batch_size,batch_size))
                                   +tf.reshape(tf.repeat(x*tf.abs(Delta_output-Delta_0),batch_size),(batch_size,batch_size))
                                   +tf.reshape(tf.repeat(y*tf.abs(Delta_output),batch_size),(batch_size,batch_size)))
        Delta_0 = tf.reshape(tf.repeat(Delta_0*(x-1),batch_size),(batch_size,batch_size))
        s = payoff_vec-u_1-u_2-Delta-Delta_0-bias+transaction_costs
        loss = tf.reduce_mean(bias) + gamma* tf.reduce_mean(tf.square(tf.nn.relu(s)))
        return loss
    
    # Define Gradient
    def grad(bias,Delta_0,model_u1_bid,model_u1_ask,model_u2_bid,model_u2_ask,model_Del, x,y):
        with tf.GradientTape() as tape:
            loss_value = loss(bias,Delta_0,model_u1_bid,model_u1_ask,model_u2_bid,model_u2_ask,model_Del,x,y)
        var_train = bias+Delta_0+model_u1_bid.trainable_variables + model_u1_ask.trainable_variables+model_u2_ask.trainable_variables+model_u2_bid.trainable_variables+model_Del.trainable_variables
        return tape.gradient(loss_value,var_train), loss_value
    
    #Define new hand-tailored layers
    class Call_option_layer_ask(layers.Layer):        
        def __init__(self, strikes = [], prices_ask = [],input_length = 1):
            super(Call_option_layer_ask, self).__init__()
            self.strikes = tf.cast(strikes,dtype = tf.float32)
            self.prices_ask = tf.cast(prices_ask,dtype = tf.float32)
            self.input_length = input_length
            self.strikes_matrix = tf.transpose(tf.reshape(tf.repeat(self.strikes,self.input_length),
                                             shape = (len(self.strikes),self.input_length)))
            self.prices_matrix = tf.transpose(tf.reshape(tf.repeat(self.prices_ask,self.input_length),
                                            shape = (len(self.prices_ask),self.input_length)))
        def call(self, inputs):
            x = tf.reshape(tf.repeat(inputs,len(self.strikes)),
                                             shape = (self.input_length,len(self.strikes)))
            x = tf.cast(x,dtype = tf.float32)
            x = tf.nn.relu(x - self.strikes_matrix)-self.prices_matrix
            return x
        
    
    class Call_option_layer_bid(layers.Layer):        
        def __init__(self, strikes = [], prices_bid = [],input_length = 1):
            super(Call_option_layer_bid, self).__init__()
            self.strikes = tf.cast(strikes,dtype = tf.float32)
            self.prices_bid = tf.cast(prices_bid,dtype = tf.float32)
            self.input_length = input_length
            self.strikes_matrix = tf.transpose(tf.reshape(tf.repeat(self.strikes,self.input_length),
                                             shape = (len(self.strikes),self.input_length)))
            self.prices_matrix = tf.transpose(tf.reshape(tf.repeat(self.prices_bid,self.input_length),
                                            shape = (len(self.prices_bid),self.input_length)))
        def call(self, inputs):
            x = tf.reshape(tf.repeat(inputs,len(self.strikes)),
                                             shape = (self.input_length,len(self.strikes)))
            x = tf.cast(x,dtype = tf.float32)
            x = tf.nn.relu(x - self.strikes_matrix)-self.prices_matrix
            return -x

    
    # # Create the Models
    model_u1_bid = tf.keras.Sequential([
        Call_option_layer_bid(strikes = strikes1,
                              prices_bid = prices1_bid,
                              input_length = batch_size),
        tf.keras.layers.Dense(1, kernel_constraint=non_neg(),use_bias=False)
        ])
    
    model_u1_ask = tf.keras.Sequential([
        Call_option_layer_ask(strikes = strikes1,
                              prices_ask = prices1_ask,
                              input_length = batch_size),
        tf.keras.layers.Dense(1, kernel_constraint=non_neg(),use_bias=False)
        ])
    
    model_u2_bid = tf.keras.Sequential([
        Call_option_layer_bid(strikes = strikes2,
                              prices_bid = prices2_bid,
                              input_length = batch_size),
        tf.keras.layers.Dense(1, kernel_constraint=non_neg(),use_bias=False)
        ])
    
    model_u2_ask = tf.keras.Sequential([
        Call_option_layer_ask(strikes = strikes2,
                              prices_ask = prices2_ask,
                              input_length = batch_size),
        tf.keras.layers.Dense(1, kernel_constraint=non_neg(),use_bias=False)
        ])
    
    model_Delta_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(1,), activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])

    # Do the Optimization
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    losses = []
    
    # Creating variables for Optimization
    bias = [tf.Variable(initial_value= [[0.]], trainable= True, dtype= tf.float32)]
    Delta_0 = [tf.Variable(initial_value= [[0.]], trainable= True, dtype= tf.float32)]
    samples = generate_samples(batch_size)
    
    # Training Loop
    for i in range(nr_iterations):
        x = next(samples)
        y = next(samples)
        grad_u, current_loss =  grad(bias,
                                     Delta_0,
                                     model_u1_bid,
                                     model_u1_ask,
                                     model_u2_bid,
                                     model_u2_ask,
                                     model_Delta_1,
                                     x,y)
        var_train = bias+Delta_0+model_u1_bid.trainable_variables + model_u1_ask.trainable_variables+model_u2_ask.trainable_variables+model_u2_bid.trainable_variables+model_Delta_1.trainable_variables
        optimizer.apply_gradients(zip(grad_u,var_train))
        losses.append(current_loss.numpy()*(1-2*lower_bound)*s0)
        if i % 100 == 0 and i > 0:
            print("Iteration:{}, Avg. Loss: {}".format((i),np.mean(losses[-100])))              
    plt.plot(range(nr_iterations-100),losses[100:])
    plt.show()
    print("Iteration result: {}".format(np.mean(losses[-(round(nr_iterations*0.05))])))

    return np.mean(losses[-(round(nr_iterations*0.1))])


mot_nn_bidask(strikes1 = np.array([100.]),
                  prices1_bid = np.array([5.]),
                  prices1_ask= np.array([8.]),
                  strikes2 = np.array([120.]),
                  prices2_bid = np.array([3.]),
                  prices2_ask= np.array([5.]),
                  s0=100,
                  lower_bound = True)
