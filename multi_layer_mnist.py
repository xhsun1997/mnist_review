import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("D:/pythonStudy/MNIST_data/",one_hot=True)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

model_save_path='D:/MNIST_practice/log/multi_layer_mnist/multi_layer_mnist.ckpt'

def train(batch_size):
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    y=tf.placeholder(dtype=tf.float32,shape=[batch_size,10])
    
    weights_1=tf.Variable(tf.random_normal(shape=[784,256],dtype=tf.float32))
    bias_1=tf.Variable(tf.zeros(shape=[256]))
    
    layer_1_out=tf.matmul(x,weights_1)+bias_1
    
    weights_2=tf.Variable(tf.random_normal(shape=[256,10],dtype=tf.float32))
    bias_2=tf.Variable(tf.zeros(shape=[10]))
    
    output=tf.matmul(layer_1_out,weights_2)+bias_2
    assert output.shape==(batch_size,10)==y.shape
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
    optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    correct_prediction=tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    assert correct_prediction.shape==(batch_size,)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    saver=tf.train.Saver()
    num_batches=int(mnist.train.images.shape[0]/batch_size)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(20):
            epoch_loss=0.0
            for i in range(num_batches):
                x_batches,y_batches=mnist.train.next_batch(batch_size)
                assert x_batches.shape==x.shape and y_batches.shape==y.shape
                _,loss_val=sess.run([optimizer,loss],feed_dict={x:x_batches,y:y_batches})
                epoch_loss+=loss_val
            print("epoch loss is",epoch_loss/num_batches)
            test_x,test_y=mnist.test.next_batch(batch_size)
            print("acc is ",sess.run(accuracy,feed_dict={x:test_x,y:test_y}))
            saver.save(sess,model_save_path)
#train(64)

def test(batch_size):
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    weights_1=tf.Variable(tf.random_normal(shape=[784,256],dtype=tf.float32))
    bias_1=tf.Variable(tf.zeros(shape=[256]))
    
    layer_1_out=tf.matmul(x,weights_1)+bias_1
    
    weights_2=tf.Variable(tf.random_normal(shape=[256,10],dtype=tf.float32))
    bias_2=tf.Variable(tf.zeros(shape=[10]))
    
    output=tf.matmul(layer_1_out,weights_2)+bias_2
    assert output.shape==(batch_size,10)
    
    what_we_what=tf.argmax(output,1)
    assert what_we_what.shape==(batch_size,)
    
    saver=tf.train.Saver()
    
    string_list=model_save_path.split('/')[:-1]
    string=""
    for s in string_list:
        string+=s
        string+="/"
    print(string)
    ckpt=tf.train.latest_checkpoint(string)
    print(ckpt)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if ckpt:
            saver.restore(sess,ckpt)
        else:
            print("path error")
            return
        x_batches,y_batches=mnist.test.next_batch(batch_size)
        assert x_batches.shape==x.shape
        predict_value,want_value=sess.run([output,what_we_what],feed_dict={x:x_batches})
        print(want_value)
        assert predict_value.shape==(batch_size,10)
        for predict,value in zip(predict_value,want_value):
            assert np.argmax(predict)==value
        assert type(y_batches)==np.ndarray==type(x_batches)
        
        target=[]
        for i in range(batch_size):
            target.append(np.argmax(y_batches[i]))
        print(target)
        
        import pylab
        for batch in range(batch_size):
            im=x_batches[batch].reshape(28,28)
            assert im.shape==(28,28)
            pylab.imshow(im)
            pylab.show()
            
test(3)
            