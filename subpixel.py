import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def _phase_shift(I, r):
    #print("I:",I,"I.get_shape():",I.get_shape().as_list())
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    #X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
   
    return tf.reshape(X, (bsize, a*r, b*r, 1))
   

def _phase_shift_1dim(I, r):
    #print("I:",I,"I.get_shape():",I.get_shape().as_list())
    #bsize, a, b, c = I.get_shape().as_list()
    bsize, a, b, c = I.shape
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    
    X = tf.reshape(I, (bsize, a, b, r, r))
    #X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    #print("Input X shape:",X.get_shape(),"scale:",r)
    if color:

        Xc = tf.split(X, 3, 3) 

        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift_1dim(X, r)
    #print("output X shape:",X.get_shape())
    return X

    
def PS_1dim(I, r):
    r = int(r)
    O = np.zeros((I.shape[0]*r, I.shape[1]*r, int(I.shape[2]/(r*r))))
    for x in range(O.shape[0]):
        for y in range(O.shape[1]):
            for c in range(O.shape[2]):
                c += 1
                a = np.floor(x/r).astype("int")
                b = np.floor(y/r).astype("int")
                d = c*r*(y%r) + c*(x%r)
                #print a, b, d
                O[x, y, c-1] = I[a, b, d]
    return O
