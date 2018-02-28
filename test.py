import tensorflow as tf
import torch

logits = tf.constant([[1,2,3,4,2],[1,2,1,1,1],[1,2,1,5,1]],dtype=tf.float64)
with tf.Session():
    nn = tf.reduce_mean(logits)
    print (nn.eval())

ll = torch.FloatTensor([[1,2,3,4,2],[1,2,1,1,1],[1,2,1,5,1]])
print( torch.mean(ll))

