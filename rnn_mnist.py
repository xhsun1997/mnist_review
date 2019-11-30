import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('D:/pythonStudy/MNIST_data/',one_hot=True)
model_save_path='D:/MNIST_practice/log/rnn_mnist/rnn_mnist.ckpt'

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

def train(batch_size,bi_direct=False):
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    y=tf.placeholder(dtype=tf.float32,shape=[batch_size,10])
    
    time_step=28
    embedding_dim=28
    hidden_dim=128
    hidden_dim1=128
    hidden_dim2=256
    x=tf.reshape(tensor=x,shape=[batch_size,time_step,embedding_dim])
    if bi_direct==False:
        cell_1=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim1)
        cell_2=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim2)
        cells=tf.contrib.rnn.MultiRNNCell([cell_1,cell_2])
        initial_state=cells.zero_state(batch_size=batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(cells,x,initial_state=initial_state)
        assert outputs.shape==(batch_size,time_step,hidden_dim2)
        assert len(states)==2==len(states[0])==len(states[1])
        assert states[0][1].shape==states[0][0].shape==(batch_size,hidden_dim1)
        assert states[1][0].shape==states[1][1].shape==(batch_size,hidden_dim2)
        outputs_t=tf.transpose(outputs,perm=[1,0,2])
        rnn_out=outputs_t[-1]
        assert rnn_out.shape==(batch_size,hidden_dim2)
        rnn_out_h=states[1][1]
        
        weights=tf.Variable(tf.random_normal(shape=[hidden_dim2,10],dtype=tf.float32))
        biases=tf.Variable(tf.zeros(shape=[10]))
        pred=tf.matmul(rnn_out,weights)+biases
        pred_2=tf.matmul(rnn_out_h,weights)+biases
        assert pred.shape==pred_2.shape==(batch_size,10)==y.shape
    else:
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,dtype=tf.float32)
        assert len(outputs)==len(states)==2
        output_fw,output_bw=outputs
        states_fw,states_bw=states
        states_fw_c,states_fw_h=states_fw
        states_bw_c,states_bw_h=states_bw
        output_bw_t=tf.transpose(output_bw,perm=[1,0,2])[-1]
        assert output_bw_t.shape==(batch_size,hidden_dim)
        assert states_bw_c.shape==states_bw_h.shape==(batch_size,hidden_dim)==states_fw_c.shape==states_fw_h.shape
        outputs_concat=tf.concat(values=[output_fw,output_bw],axis=-1)
        outputs_concat_t=tf.transpose(outputs_concat,perm=[1,0,2])
        lstm_out=outputs_concat_t[-1]
        lstm_out_2=tf.concat(values=[states_fw_h,output_bw_t],axis=-1)
        assert lstm_out.shape==(batch_size,2*hidden_dim)==lstm_out_2.shape
        
        weights=tf.Variable(tf.random_normal(shape=[2*hidden_dim,10],dtype=tf.float32))
        biases=tf.Variable(tf.zeros(shape=[10]))
        pred=tf.matmul(lstm_out,weights)+biases
        pred_2=tf.matmul(lstm_out_2,weights)+biases
        assert pred.shape==pred_2.shape==(batch_size,10)
        
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    saver=tf.train.Saver()
    num_batches=int(mnist.train.images.shape[0]/batch_size)#55000/64
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            total_loss=0.0
            for i in range(num_batches):
                x_batches,y_batches=mnist.train.next_batch(batch_size)
                x_batches=np.reshape(x_batches,(batch_size,time_step,embedding_dim))
                feed_dict={x:x_batches,y:y_batches}
                pred_value,pred_2_value,loss_value,_=sess.run([pred,pred_2,loss,optimizer],feed_dict=feed_dict)
                assert pred_value.shape==pred_2_value.shape==(batch_size,10)
                for i in range(batch_size):
                    for j in range(10):
                        assert pred_2_value[i][j]==pred_value[i][j]#也就是说明 states[-1][-1]即为RNN的最后输出 等价于outputs.transpose([1,0,2])[-1]
                total_loss+=loss_value
            print("average loss is ",total_loss/num_batches)
            x_test,y_test=mnist.test.next_batch(batch_size)
            x_test=x_test.reshape((batch_size,time_step,embedding_dim))
            assert x_test.shape==x.shape and y_test.shape==y.shape
            print("accuracy is ",accuracy.eval(feed_dict={x:x_test,y:y_test}))
            saver.save(sess,model_save_path)
#train(64,bi_direct=True)

def test(batch_size,bi_direct=False):
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,784])
    y=tf.placeholder(dtype=tf.float32,shape=[batch_size,10])
    
    time_step=28
    embedding_dim=28
    hidden_dim=128
    hidden_dim1=128
    hidden_dim2=256
    x=tf.reshape(tensor=x,shape=[batch_size,time_step,embedding_dim])
    if bi_direct==False:
        cell_1=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim1)
        cell_2=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim2)
        cells=tf.contrib.rnn.MultiRNNCell([cell_1,cell_2])
        initial_state=cells.zero_state(batch_size=batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(cells,x,initial_state=initial_state)
        assert outputs.shape==(batch_size,time_step,hidden_dim2)
        assert len(states)==2==len(states[0])==len(states[1])
        assert states[0][1].shape==states[0][0].shape==(batch_size,hidden_dim1)
        assert states[1][0].shape==states[1][1].shape==(batch_size,hidden_dim2)
        outputs_t=tf.transpose(outputs,perm=[1,0,2])
        rnn_out=outputs_t[-1]
        assert rnn_out.shape==(batch_size,hidden_dim2)
        rnn_out_h=states[1][1]
        
        weights=tf.Variable(tf.random_normal(shape=[hidden_dim2,10],dtype=tf.float32))
        biases=tf.Variable(tf.zeros(shape=[10]))
        pred=tf.matmul(rnn_out,weights)+biases
        pred_2=tf.matmul(rnn_out_h,weights)+biases
        assert pred.shape==pred_2.shape==(batch_size,10)==y.shape
        
    
    else:
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,dtype=tf.float32)
        assert len(outputs)==len(states)==2
        output_fw,output_bw=outputs
        states_fw,states_bw=states
        states_fw_c,states_fw_h=states_fw
        states_bw_c,states_bw_h=states_bw
        output_bw_t=tf.transpose(output_bw,perm=[1,0,2])[-1]
        assert output_bw_t.shape==(batch_size,hidden_dim)
        assert states_bw_c.shape==states_bw_h.shape==(batch_size,hidden_dim)==states_fw_c.shape==states_fw_h.shape
        outputs_concat=tf.concat(values=[output_fw,output_bw],axis=-1)
        outputs_concat_t=tf.transpose(outputs_concat,perm=[1,0,2])
        lstm_out=outputs_concat_t[-1]
        lstm_out_2=tf.concat(values=[states_fw_h,output_bw_t],axis=-1)
        assert lstm_out.shape==(batch_size,2*hidden_dim)==lstm_out_2.shape
        
        weights=tf.Variable(tf.random_normal(shape=[2*hidden_dim,10],dtype=tf.float32))
        biases=tf.Variable(tf.zeros(shape=[10]))
        pred=tf.matmul(lstm_out,weights)+biases
        pred_2=tf.matmul(lstm_out_2,weights)+biases
        assert pred.shape==pred_2.shape==(batch_size,10)     
          
        
        
    saver=tf.train.Saver()
    string=""
    string_list=model_save_path.split('/')[:-1]
    for s in string_list:
        string+=s
        string+="/"
    ckpt=tf.train.latest_checkpoint(string)#sting是检查点所在文件路径的上一个目录
    print(ckpt)
    predict_value=tf.argmax(pred_2,1)
    assert predict_value.shape==(batch_size,)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer()) 
        if ckpt:
            saver.restore(sess,ckpt)
        else:
            print("model save path error")
            return
        x_test,y_test=mnist.test.next_batch(batch_size)
        x_test=x_test.reshape((batch_size,time_step,embedding_dim))
        predict_val,pre_val,pre_2_val=sess.run([predict_value,pred,pred_2],feed_dict={x:x_test})
        assert predict_val.shape==(batch_size,)
        assert pre_val.shape==pre_2_val.shape==(batch_size,10)
        print(predict_val)
        target=[]
        for i in range(batch_size):
            target.append(np.argmax(y_test[i]))
        print(target)
        import pylab
        for i in range(batch_size):
            for j in range(10):
                assert pre_val[i][j]==pre_2_val[i][j]
        for i in range(batch_size):
            im=x_test[i]
            assert im.shape==(28,28)
            pylab.imshow(im)
            pylab.show()
test(5,bi_direct=True)
            
   