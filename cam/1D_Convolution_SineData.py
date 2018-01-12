# -*- coding: utf-8 -*-
'''--------------------------------------------------------------------------
 Junhong Kim
 Ph.D. Student in Industrial Management Engineering
 Korea University, Seoul, Republic of Korea
 Mobile Phone +82 10 3099 3004
 E-mail    junhongkim@korea.ac.kr
 Data Science and Business Analytics Lab
 Lab Homepage http://dsba.korea.ac.kr
--------------------------------------------------------------------------'''
#########################################################
#####################################
# For escape early stop error ( CTRL + C )
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
#########################################################



'''Raw data infile'''
Raw_Data=np.load(os.getcwd()+"/Toy_Dataset_Sine.npy")

# 크기 확인 / 100개의 Rawdata와 2개의 target(Dummy variable)을 가지고 있다.
np.shape(Raw_Data)

## 지금은 정렬되어 있으니 한번 shuffle하고 data partition을 해보도록 하자 (7:3)으로 partition하자
# 전체 데이터를 한번 shuffle하도록 하자.


#이제 7:3으로 data partition을 해보도록 하자.
def Data_Partition(Data,Train_Ratio,Target_Column):
    Full_Index = np.arange((np.shape(Data)[0]))
    np.random.shuffle(Full_Index)
    Raw_Data2=Raw_Data[Full_Index]

    Train_Data = Raw_Data2[:np.int((np.floor(np.shape(Data)[0]*Train_Ratio)))]
    Test_Data = Raw_Data2[np.int((np.floor(np.shape(Data)[0]*Train_Ratio))):]
    Train_Target = Train_Data[:,Target_Column]
    Test_Target = Test_Data[:,Target_Column]
    Train_Input = Train_Data[:,:Target_Column[0]]
    Test_Input = Test_Data[:,:Target_Column[0]]
    return(Train_Input,Train_Target,Test_Input,Test_Target) # end function


Train_Input,Train_Target,Test_Input,Test_Target = Data_Partition(Raw_Data,0.7,range(100,102))

# shape를 다시 한번 확인 하도록 하자. (확인차)
print('------------------------------\n'+
'     Check of Dataset Size! \n'+
'------------------------------\n'+
"Train_Input_Size : %s" % (np.shape(Train_Input),) + '\n'+
"Test_Input_Size : %s" % (np.shape(Test_Input),) + '\n'+
"Train_Target_Size : %s" % (np.shape(Train_Target),) + '\n'+
"Train_Target_Size : %s" % (np.shape(Test_Target),))

'''

plt.figure(figsize=(2, 1))
x = range(100)
y = list(chain.from_iterable(Train_Input[:1]))
y2 = list(chain.from_iterable(Test_Input[:1]))
# Plotting each class first row
for i in np.arange(2):
    plt.subplot(2, 1, i + 1)
    if (i==0):
        plt.plot(x, y)
        plt.ylim((-2, 2))
    if (i==1):
        plt.plot(x, y2)
        plt.ylim((-2, 2))
        plt.show()
'''

'''자이제 1D convolution을 진행하고 CAM까지 진행해 보도록 하자'''
'''네트워크를 건설하자'''
#########################################################
input_height = 1 # 1D-shape (1)
input_width = 100 # 1D-shape (2)
num_labels = 2
num_channels = 1
Train_Input=Train_Input.reshape(np.shape(Train_Input)[0],input_height,input_width,num_channels)
Test_Input=Test_Input.reshape(np.shape(Test_Input)[0],input_height,input_width,num_channels)
#########################################################


np.shape(Train_Input)
np.shape(Test_Input)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')

# define He initialization function
def He_Init(K_W, K_H, I_F, O_F):
    return tf.Variable(tf.truncated_normal([K_W, K_H, I_F, O_F], stddev=1, mean=0) /
                       (tf.sqrt(K_W * K_H * I_F / 2)), dtype=tf.float32)





# Placeholder setting
X = tf.placeholder(tf.float32, [None, np.shape(Train_Input)[1],np.shape(Train_Input)[2],np.shape(Train_Input)[3]])
Y = tf.placeholder(tf.float32, [None,2])
Learning_Rate = tf.placeholder(tf.float32)
train_bool = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)
print("PlaceHolder complete!")
# One_Hot encoding based on target variable


with tf.name_scope('Conv_1_Set'):
    # Before Encoder Cell convolution
    Weight1 = He_Init(1, 3, np.shape(Train_Input)[3], 128)
    After_Conv1 = tf.nn.relu(
        tf.layers.batch_normalization(tf.add(tf.nn.conv2d(X, Weight1, strides=[1, 1, 1, 1], padding='SAME'),bias_variable([128])), momentum=0.9,
                                      training=train_bool),name="Conv1")

    Weight2 = He_Init(1, 3, 128, 128)
    After_Conv2 = tf.nn.relu(
        tf.layers.batch_normalization(tf.add(tf.nn.conv2d(After_Conv1, Weight2, strides=[1, 1, 1, 1], padding='SAME'),bias_variable([128])),
                                      momentum=0.9, training=train_bool),name="Conv2")

    After_Conv2_Pool = apply_max_pool(After_Conv2, 2, 2)


with tf.name_scope('Conv_2_Set'):
    Weight3 = He_Init(1, 3, 128, 256)
    After_Conv3 = tf.nn.relu(
        tf.layers.batch_normalization(tf.add(tf.nn.conv2d(After_Conv2_Pool, Weight3, strides=[1, 1, 1, 1], padding='SAME'),bias_variable([256])), momentum=0.9,
                                      training=train_bool),name="Conv3")
    Weight4 = He_Init(1, 3,256, 256)
    After_Conv4 = tf.nn.relu(
        tf.layers.batch_normalization(tf.add(tf.nn.conv2d(After_Conv3, Weight4, strides=[1, 1, 1, 1], padding='SAME'),bias_variable([256])),
                                      momentum=0.9, training=train_bool),name="Conv4")



with tf.name_scope('Global_Avg_Polling'):
####################################################################################################################
##### Average Pooling
####################################################################################################################
    TKS=After_Conv4.get_shape().as_list()[2] # The Kernel Size for Global average pooling
    Global_Avg_Pool=tf.cast(tf.nn.avg_pool(After_Conv4,[1,TKS,TKS,1],[1,TKS,TKS,1],padding='SAME',name='Average_Pool'),tf.float32)
    shape = Global_Avg_Pool.get_shape().as_list()
    c_flat = tf.reshape(Global_Avg_Pool, [-1, shape[1] * shape[2] * shape[3]])



with tf.name_scope('FC_Network'):
    Conv1x1_W = He_Init(1, 1, 256, 2)
    After_Conv1x1 = tf.nn.conv2d(Global_Avg_Pool, Conv1x1_W, strides=[1, 1, 1, 1], padding='SAME')
    Conv1x1_Bias = tf.Variable(tf.zeros([2]))
    Logits = tf.reshape(After_Conv1x1 + Conv1x1_Bias,[-1, 2])


with tf.name_scope('Opt'):
    Real_Softmax=tf.nn.softmax(Logits)
    Index_Max = tf.argmax(Real_Softmax, axis=1)
    SoftMax=tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Logits)
    loss = tf.reduce_mean(SoftMax)
    tf.summary.scalar('Loss', loss)
    # optimizer
    optimizer = tf.train.RMSPropOptimizer(Learning_Rate)
    train = optimizer.minimize(loss)
    # Session start
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # Append batch_norm parameter during training step
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("Optimization method complete!")


# train / RMSPROP
# Summary 정의
Tensorboard_Root_Path = os.getcwd()+"/CNN_Tensorboard/"
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(Tensorboard_Root_Path + '/train', sess.graph)
test_writer = tf.summary.FileWriter(Tensorboard_Root_Path + '/test')
print("Session Ready! & Let's Train!")
###################################################################################################################################


batch_size=32
Num_of_Iterlation_Based_On_Epoch=np.shape(Train_Input)[0]//batch_size
Num_of_Epoch=10

for z in range(Num_of_Epoch):
    for i in range(Num_of_Iterlation_Based_On_Epoch):
        TR_Index = np.random.choice(np.shape(Train_Input)[0], 32, replace=False)
        _, __ = sess.run([train, extra_update_ops],
                         feed_dict={X: Train_Input[TR_Index, :, :, :],
                                    Y: Train_Target[TR_Index,:],
                                    Learning_Rate: 0.00001, train_bool: True,
                                    keep_prob: 1})
        if (i % 10 == 0):
            TR_Index = np.random.choice(np.shape(Train_Input)[0], 32, replace=False)
            summary_tr = sess.run(merged,
                                  feed_dict={X: Train_Input[TR_Index, :, :, :],
                                             Y: Train_Target[TR_Index,:],
                                             Learning_Rate: 0.00001, train_bool: False,
                                             keep_prob: 1})

            TE_Index = np.random.choice(np.shape(Test_Input)[0], 32, replace=False)
            summary_te = sess.run(merged,
                                  feed_dict={X: Test_Input[TE_Index, :, :, :],
                                             Y: Test_Target[TE_Index,:],
                                             Learning_Rate: 0.00001, train_bool: False,
                                             keep_prob: 1})

            train_writer.add_summary(summary_tr, z*Num_of_Iterlation_Based_On_Epoch+i)
            test_writer.add_summary(summary_te, z*Num_of_Iterlation_Based_On_Epoch+i)
            print(z*Num_of_Iterlation_Based_On_Epoch+i)

    print("Epoch:"+str(z+1))







def Visualization():
    TE_Index = np.random.choice(np.where(Test_Target[:, 1] == 1)[0], 1, replace=False)
    ActivationMap, Linear_Comb_Weight = sess.run([After_Conv4, Conv1x1_W], feed_dict={X: Test_Input[TE_Index, :, :, :],
                                                                                      Y: Test_Target[TE_Index, :],
                                                                                      Learning_Rate: 0.00001,
                                                                                      train_bool: False,
                                                                                      keep_prob: 1})
    Reshape_Act_Map=np.reshape(ActivationMap,[50,256])
    Act_Weights = np.reshape(Linear_Comb_Weight[:,:,:,1], [256])
    CAM_Results_Prior=list()
    for  i in range(np.shape(Reshape_Act_Map)[1]):
        CAM_Results_Prior.append((Reshape_Act_Map)[:,i]*Act_Weights[i])
    First_CAM=np.sum(CAM_Results_Prior,axis=0)
    MINMAX_CAM_Score = (First_CAM - np.min(First_CAM))/(np.max(First_CAM) - np.min(First_CAM))
    plt.figure(figsize=(2, 1))
    x = range(100)
    y = np.reshape(Test_Input[TE_Index, :, :, :],100)
    x2 = range(50)
    y2 = MINMAX_CAM_Score
    # Plotting each class first row
    for i in np.arange(2):
        plt.subplot(2, 1, i + 1)
        if (i == 0):
            plt.plot(x, y)
            plt.ylim((-2, 2))
        if (i == 1):
            plt.plot(x2, y2)
            plt.ylim((-2, 2))
            plt.show()

# 마지막 Visualization으로 해당 모델의 Class Activation MAP을 확인해 보자!
for iter in range(10):
    Visualization()















