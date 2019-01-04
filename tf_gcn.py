import tensorflow as tf
import numpy as np
from toy_dataset import ToyDataset
from gcn_util import GCNUtil
import random
import math

tds = ToyDataset()
gcnu = GCNUtil()

dsLabels = tds.get_label()
dsGraph = tds.get_graph()
dsAdjacency = []
dsFeatures = []
dsDegree = []

mSize = 16
for i in range(16):
  g = gcnu.get_subgraph(dsGraph, i+1, 3)
  A = gcnu.adjacency(g)
  F = gcnu.feature(A)
  D = gcnu.degree(g)
  A = A.reshape([1,A.shape[0], A.shape[1]])
  F = F.reshape([1,F.shape[0], F.shape[1]])
  D = D.reshape([1,D.shape[0], D.shape[1]])
  
  if i == 0:
    dsAdjacency = A
    dsDegree = D
    dsFeatures = F
  else:
    dsAdjacency = np.concatenate([dsAdjacency, A])
    dsDegree = np.concatenate([dsDegree, D])
    dsFeatures = np.concatenate([dsFeatures, F])

print("-"*30)
print("Shape information")
print("-"*30)
print("adjacency", dsAdjacency.shape)
print("features ", dsFeatures.shape)
print("degree   ", dsDegree.shape)
print("labels   ", dsLabels.shape)
print("-"*30)

print()
print()

print("-"*30)
print("Build model")
print("-"*30)
features = tf.placeholder(tf.float32, shape=(mSize,mSize))
adjacency = tf.placeholder(tf.float32, shape=(mSize,mSize))
degree = tf.placeholder(tf.float32, shape=(mSize,mSize))
labels = tf.placeholder(tf.float32, shape=(6))


weights = tf.Variable(tf.random_normal([mSize,mSize], stddev=1))


def GCN(features, adjacency, degree, weights):
    with tf.name_scope('gcn_layer'):
        d_ = tf.pow(degree + tf.eye(mSize), +1.0)
        y = tf.matmul(d_, tf.matmul(adjacency, d_))
        kernel = tf.matmul(features, weights)

        return tf.nn.relu(tf.matmul(y, kernel))

gcn = GCN(features, adjacency, degree, weights)

model = tf.reshape(gcn, shape=[-1, mSize, mSize])
model = tf.contrib.layers.flatten(model)
model = tf.layers.dense(model, 6)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        - labels * tf.nn.log_softmax(model)
    )
"""
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = softmax, labels = labels
        )
    )
"""
train_op = tf.train.AdamOptimizer(0.0005, 0.9).minimize(loss)
print()
print()    
print("-"*30)
print("Training started")
print("-"*30)

initializer = tf.initializers.global_variables()
with tf.Session() as sess:
    sess.run(initializer)
    
    for i in range(1000):
        rndId = random.randint(0,15)
        # remove improper classification category=2
        while np.argmax(dsLabels[rndId], axis=0) == 2 :
            rndId = random.randint(0,15)
        _, lossVal = sess.run([train_op, loss], feed_dict = {
                features: dsFeatures[rndId],
                adjacency: dsAdjacency[rndId],
                degree: dsDegree[rndId] ,
                labels: dsLabels[rndId]
            }
        )

        print("%d\t%2.6f"% (i, lossVal))

