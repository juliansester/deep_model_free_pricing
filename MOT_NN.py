# -*- coding: utf-8 -*-
"""
MOT-Neural Network Approximation
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import layers, Model,regularizers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def mot_nn(minimize = False, batch_size = 2**8,nr_iterations = 500,gamma = 40):
    def payoff(x,y):
        #return tf.constant(5)
        return tf.abs(x-y)*(1-2*minimize)
    
    # Define the Loss Function
    def loss(model1,model2,model_Del,x,y):
        u_1 = model1(x)
        u_2 = model2(y)
        Ints = tf.reduce_mean(u_1)+tf.reduce_mean(u_2)
        u_1 = tf.reshape(tf.repeat(u_1,batch_size),(batch_size,batch_size))
        u_2 = tf.transpose(tf.reshape(tf.repeat(u_2,batch_size),(batch_size,batch_size)))
        Delta = tf.reshape(tf.repeat(model_Del(x),batch_size),(batch_size,batch_size))
        Delta = tf.math.multiply(Delta,tf.transpose(tf.reshape(tf.repeat(y,batch_size),(batch_size,batch_size)))-tf.reshape(tf.repeat(x,batch_size),(batch_size,batch_size)))
        payoff_vec = payoff(x,tf.transpose(y))
        s = payoff_vec-u_1-u_2-Delta
        loss = Ints + gamma* tf.reduce_mean(tf.square(tf.nn.relu(s)))
        return loss
    
    # Define Gradient
    def grad(model1,model2,model_Del, x,y):
        with tf.GradientTape() as tape:
            loss_value = loss(model1,model2,model_Del,x,y)
        variables_totrain = model_u1.trainable_variables+model_u2.trainable_variables+model_Delta_1.trainable_variables
        return tape.gradient(loss_value,variables_totrain), loss_value
    
    # # Create the Models
    model_u1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(1,), activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])
    
    model_u2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(1,), activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])
    
    model_Delta_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(1,), activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])
    
    
    # Do the Optimization
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    losses = []

    for i in range(nr_iterations):
        x = tf.random.uniform(shape = [batch_size,1])*2-1
        y = tf.random.uniform(shape = [batch_size,1])*4-2
        grad_u, current_loss =  grad(model_u1,model_u2,model_Delta_1, x,y)
        optimizer.apply_gradients(zip(grad_u,model_u1.trainable_variables+model_u2.trainable_variables+model_Delta_1.trainable_variables))
        losses.append(current_loss.numpy()*(1-2*minimize))
        if i % 100 == 0 and i > 0:
            print("Iteration:{}, Avg. Loss: {}".format((i),np.mean(losses[-100])))
    plt.plot(range(nr_iterations),losses)
    print("Iteration result: {}".format(np.mean(losses[-(round(nr_iterations*0.05))])))
    return np.mean(losses[-(round(nr_iterations*0.1))])


mot_nn(minimize = True)
