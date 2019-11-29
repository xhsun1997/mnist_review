import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("D:/pythonStudy/MNIST_data/",one_hot=True)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

model_save_path='D:/MNIST_practice/log/cnn_mnist/cnn_mnist.ckpt'
def train_conv(batch_size):

    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    y=tf.placeholder(dtype=tf.float32,shape=[batch_size,10])
    conv_input=tf.reshape(tensor=x,shape=[batch_size,28,28,1])
    weights_1=tf.Variable(tf.random_normal(shape=[5,5,1,32],dtype=tf.float32))
    conv_1=tf.nn.relu(tf.nn.conv2d(conv_input,weights_1,strides=[1,2,2,1],padding="SAME"))
    assert conv_1.shape==(batch_size,14,14,32)
    pool_1=tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    assert pool_1.shape==(batch_size,7,7,32)
    weights_2=tf.Variable(tf.random_normal(shape=[3,3,32,64],dtype=tf.float32))
    conv_2=tf.nn.relu(tf.nn.conv2d(pool_1,weights_2,strides=[1,1,1,1],padding="VALID"))
    assert conv_2.shape==(batch_size,5,5,64)
    conv_output=tf.reshape(conv_2,shape=[batch_size,5*5*64])
    output=tf.contrib.layers.fully_connected(conv_output,num_outputs=10,activation_fn=None)
    assert output.shape==(batch_size,10)==y.shape

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
    train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    saver=tf.train.Saver(max_to_keep=2)
    num_batches=int(mnist.train.images.shape[0]/batch_size)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            epoch_loss=0.0
            for i in range(num_batches):
                x_batches,y_batches=mnist.train.next_batch(batch_size)
                assert x_batches.shape==x.shape and y_batches.shape==y.shape
                _,loss_val=sess.run([train_op,loss],feed_dict={x:x_batches,y:y_batches})
                epoch_loss+=loss_val

            print("epoch loss is",epoch_loss/num_batches)
            test_x,test_y=mnist.test.next_batch(batch_size)
            print("acc is ",sess.run(accuracy,feed_dict={x:test_x,y:test_y}))
            saver.save(sess,model_save_path)
#train_conv(64)

def test(batch_size):
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    y=tf.placeholder(dtype=tf.float32,shape=[batch_size,10])
    conv_input=tf.reshape(tensor=x,shape=[batch_size,28,28,1])
    weights_1=tf.Variable(tf.random_normal(shape=[5,5,1,32],dtype=tf.float32))
    conv_1=tf.nn.relu(tf.nn.conv2d(conv_input,weights_1,strides=[1,2,2,1],padding="SAME"))
    assert conv_1.shape==(batch_size,14,14,32)
    pool_1=tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    assert pool_1.shape==(batch_size,7,7,32)
    weights_2=tf.Variable(tf.random_normal(shape=[3,3,32,64],dtype=tf.float32))
    conv_2=tf.nn.relu(tf.nn.conv2d(pool_1,weights_2,strides=[1,1,1,1],padding="VALID"))
    assert conv_2.shape==(batch_size,5,5,64)
    conv_output=tf.reshape(conv_2,shape=[batch_size,5*5*64])
    output=tf.contrib.layers.fully_connected(conv_output,num_outputs=10,activation_fn=None)
    assert output.shape==(batch_size,10)==y.shape
    correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    saver=tf.train.Saver(max_to_keep=2)
    what_we_what=tf.argmax(output,1)
    assert what_we_what.shape==(batch_size,)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,model_save_path)
        x_batch,y_batch=mnist.test.next_batch(batch_size)

        predict_value,want_value=sess.run([output,what_we_what],feed_dict={x:x_batch})
        assert predict_value.shape==y_batch.shape
        print(want_value)
        true_value=np.argmax(y_batch,1)
        print(true_value)
        import pylab
        for i in range(batch_size):
            im=x_batch[i]
            im=im.reshape(28,28)
            pylab.imshow(im)
            pylab.show()
test(3)