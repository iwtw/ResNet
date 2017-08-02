import numpy as np
import tensorflow as tf
import utils
import numpy as np
import data_input
import tensorflow.contrib.layers
import time
import os 
N_GPUS = 2
if N_GPUS == 0 :
    DEVICES = ["/cpu:0"]
else : DEVICES = ["/gpu:{}".format(i) for i in range(N_GPUS)]

FILE_DATA = ""
DIM = 64
BATCH_SIZE = 64
LABEL_NUM = 10575 
N_EPOCHS = 20
EPOCH_SIZE = int ( 442839 / BATCH_SIZE )
FILE_LIST = [ "/mnt/tfrecord/webface-lcnn-40-new/train_00{}.tfrecord".format( i ) for i in range( 10 ) ]
LEARNING_RATES = [ 1e-3 , 5e-3 , 1e-4 , 5e-5 , 1e-5 ]
CHECKPOINT_PATH="checkpoints/resnet_34"
def build_graph():
    file_queue = tf.train.string_input_producer( FILE_LIST ) 
    images , labels = data_input.get_batch( file_queue , (112,96) , BATCH_SIZE , n_threads = 4 , min_after_dequeue = 0 , flip_flag = True )

    with tf.device( "/gpu:0" ):
        labels = tf.one_hot( labels , LABEL_NUM )
        images_split = tf.split( images , len(DEVICES) , axis = 0 )
        labels_split = tf.split( labels , len(DEVICES) , axis = 0 )
        losses = []
    
    for device_idx , device in enumerate( DEVICES ):
        with tf.device( device ):
            conv1 = utils.conv2d( images_split[device_idx] , outputs_dim = 64 , kernel_size = 7 , stride = 2 , padding="SAME" , he_init = False , activation_fn = None  )
            conv2_x = conv1
            conv2_x = tf.contrib.layers.max_pool2d(conv2_x , kernel_size = 3 , stride = 2 )
            for i in range(3):
                conv2_x = utils.basic_block( conv2_x , outputs_dim = DIM , kernel_size = 3 , stride = 1  )
            conv3_x = conv2_x
            for i in range(4):
                conv3_x = utils.basic_block( conv3_x , outputs_dim = DIM*2**1 , kernel_size = 3 , stride = 2  )
            conv4_x = conv3_x
            for i in range(6):
                conv4_x = utils.basic_block( conv4_x , outputs_dim =  DIM*2**2 , kernel_size = 3 , stride = 2  )
            conv5_x = conv4_x
            for i in range(3):
                conv5_x = utils.basic_block( conv5_x , outputs_dim = DIM*2**3 , kernel_size = 3 , stride = 2  )
            avg_pool =  tf.contrib.layers.avg_pool2d( conv5_x , kernel_size = 1 , stride = 1  )
            fc = utils.fully_connected( avg_pool , outputs_dim = LABEL_NUM , he_init = False , activation_fn = None  )
            loss = tf.losses.softmax_cross_entropy( onehot_labels = labels_split[device_idx]  , logits = fc )
            losses.append(loss)
    loss = tf.add_n( losses ) * ( 1.0 / len(DEVICES)) * ( 1.0 / BATCH_SIZE ) 
    tf.summary.scalar("loss",loss)
    return fc , loss

def train():
    _ , loss = build_graph()
    merged_summary = tf.summary.merge_all()

    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable = False , name = "global_step" )
    boundaries = [ int(N_EPOCHS/5) * EPOCH_SIZE , int(N_EPOCHS/5)*2 * EPOCH_SIZE , int(N_EPOCHS/5)*3 * EPOCH_SIZE , int(N_EPOCHS/5)*4 * EPOCH_SIZE ]
    lr = tf.train.piecewise_constant( global_step , boundaries  , LEARNING_RATES )
    update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer( learning_rate = lr).minimize( loss , global_step = global_step ) 
    config = tf.ConfigProto( allow_soft_placement = True , log_device_placement = False  )
    config.gpu_options.allow_growth = True
    with tf.Session( config = config ) as sess:
        saver = tf.train.Saver( var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) ) 
        if os.path.exists( CHECKPOINT_PATH+"/resnet_34.meta" ):
            saver.restore( sess = sess , save_path =  CHECKPOINT_PATH+"/resnet_34" )
        else:
            sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        writer = tf.summary.FileWriter( "summarylog" ,  sess.graph )
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess)


        it = global_step.eval
        while it() < EPOCH_SIZE * N_EPOCHS:
            if it() == 0 or it() % EPOCH_SIZE == EPOCH_SIZE - 1  or it() % 50 == 49 or it() < 10 :
                saver.save( sess = sess , save_path = CHECKPOINT_PATH+"/resnet_34"  )
                loss_ , log  = sess.run([loss , merged_summary])
                print("{} , iter {} , loss {}".format( time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())) , it() , loss_ ) )
                writer.add_summary( log , it() )
                
            sess.run(train_op)
        coord.request_stop()
        coord.join(threads)


def main(_):
    train()

if __name__=="__main__":
    tf.app.run(main)
