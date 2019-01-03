import tensorflow as tf
import numpy as np
from toy_dataset import ToyDataset
from gcn_util import GCNUtil
import random

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

print("-"*20)
print("Shape information")
print("-"*20)
print("adjacency", dsAdjacency.shape)
print("features ", dsFeatures.shape)
print("degree   ", dsDegree.shape)
print("labels   ", dsLabels.shape)
print("-"*20)

print()
print()

features = tf.placeholder(tf.float32, shape=(mSize,mSize))
adjacency = tf.placeholder(tf.float32, shape=(mSize,mSize))
degree = tf.placeholder(tf.float32, shape=(mSize,mSize))
labels = tf.placeholder(tf.float32, shape=(6))


weights = tf.Variable(tf.random_normal([mSize,mSize], stddev=1))

def layer(features, adjacency, degree, weights):
    with tf.name_scope('gcn_layer'):
        d_ = tf.pow(degree + tf.eye(mSize), -0.5)
        y = tf.matmul(d_, tf.matmul(adjacency, d_))
        kernel = tf.matmul(features, weights)

        return tf.nn.relu(tf.matmul(y, kernel))

m0 = layer(features, adjacency, degree, weights)
model = tf.reshape(m0, shape=[-1, mSize, mSize, 1])
model = tf.layers.conv2d(model, 1, kernel_size=2, activation=tf.nn.relu)
model = tf.contrib.layers.flatten(model)
model = tf.layers.dense(model, 6)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits = model, labels = labels
        )
    )
    train_op = tf.train.AdamOptimizer(0.0005, 0.9).minimize(loss)
    
print("-"*20)
print("Training started")
print("-"*20)

intializer = tf.initializers.global_variables()
with tf.Session() as sess:
    sess.run(intializer)
    
    for i in range(500):
        rnd = random.randint(0,15)
        # remove improper classification category=2
        while np.argmax(dsLabels[rnd], axis=0) == 2 :
            rnd = random.randint(0,15)
        _, lossVal, mVal = sess.run([train_op, loss, m0], feed_dict = {
                features: dsFeatures[rnd],
                adjacency: dsAdjacency[rnd],
                degree: dsDegree[rnd] + 0.05,
                labels: dsLabels[rnd]
            }
        )
        print("%d\t%2.6f"% (i, lossVal))

