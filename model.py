import scipy.misc
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import array
import time
import math
from datetime import timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



tf.__version__ ##tf1.8


# 超参数


batch_size = 200
L2NormConst = 0.001
learning_rate = 1e-4
keep_prob = tf.placeholder(tf.float32) #drop_out参数




#权重
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

#偏置
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

#卷积层
def conv_relu(x, W, stride, b):
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID') + b)

#全连接层
def matmul_relu(fc, W, b):
    return tf.nn.relu(tf.matmul(fc, W) + b)

with tf.name_scope('input_data'):
    image = tf.placeholder(tf.float32, shape=[None, 66, 200, 3]) #输入为RGB图片
    angle = tf.placeholder(tf.float32, shape=[None, 1]) #图片对应的转向角

with tf.name_scope('convlayer'):
    
    #第一层卷积
    W_conv1 = weight([5, 5, 3, 24])
    b_conv1 = bias([24])
    conv1 = conv_relu(image, W_conv1, 2, b_conv1)
    
    #第二层卷积
    W_conv2 = weight([5, 5, 24, 36])
    b_conv2 = bias([36])
    conv2 = conv_relu(conv1, W_conv2, 2, b_conv2)

    #第三层卷积
    W_conv3 = weight([5, 5, 36, 48])
    b_conv3 = bias([48])
    conv3 = conv_relu(conv2, W_conv3, 2, b_conv3)

    #第四层卷积
    W_conv4 = weight([3, 3, 48, 64])
    b_conv4 = bias([64])
    conv4 = conv_relu(conv3, W_conv4, 1, b_conv4)
    
    #第五层卷积
    W_conv5 = weight([3, 3, 64, 64])
    b_conv5 = bias([64])
    conv5 = conv_relu(conv4, W_conv5, 1, b_conv5)

with tf.name_scope('full_connected_layer'):
    
    #展开成1维张量
    conv5_flat = tf.reshape(conv5, [-1, 1152])
    
    #第一层全连接层（论文是将第二层开始视为全连接层）
    W_fc1 = weight([1152, 1164])
    b_fc1 = bias([1164])
    fc1 = matmul_relu(conv5_flat, W_fc1, b_fc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    #第二层全连接层
    W_fc2 = weight([1164, 100])
    b_fc2 = bias([100])
    fc2 = matmul_relu(fc1, W_fc2, b_fc2)
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

    #第三层全连接层
    W_fc3 = weight([100, 50])
    b_fc3 = bias([50])
    fc3 = matmul_relu(fc2, W_fc3, b_fc3)
    fc3_drop = tf.nn.dropout(fc3, keep_prob)

    #第四层全连接层
    W_fc4 = weight([50, 10])
    b_fc4 = bias([10])
    fc4 = matmul_relu(fc3, W_fc4, b_fc4)
    fc4_drop = tf.nn.dropout(fc4, keep_prob)

with tf.name_scope('output'):
    W_fc5 = weight([10, 1])
    b_fc5 = bias([1])
    angle_pred = tf.multiply(tf.atan(tf.matmul(fc4_drop, W_fc5) + b_fc5), 6)

with tf.name_scope('loss'):
    train_vars = tf.trainable_variables()
    loss = tf.reduce_mean(tf.square(tf.subtract(angle_pred, angle)))     + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst #应对过拟合


## 载入数据
images = []
angles = []

#记录batch的位置
train_batch_pointer = 0
test_batch_pointer = 0

#读取文件
with open("driving_dataset/data.txt") as f:
    for line in f:
        images.append("driving_dataset/" + line.split()[0])   #文件名
        angles.append(float(line.split()[1]) * math.pi / 180) #将角度转为弧度


num_images = len(images)

combine = list(zip(images, angles)) #两个列表->元组->一个列表

#使用随机数种子保存shuffle的状态
random.seed(1)
random.shuffle(combine)

images, angles = zip(*combine) #一个列表->按一一对应的关系恢复为两个列表

train_images = images[:int(num_images * 0.8)] #训练用图片
train_angles = angles[:int(num_images * 0.8)]  #训练图片对应的角度

test_images = images[-int(num_images * 0.2):] #测试用
test_angles = angles[-int(num_images * 0.2):]

num_train_images = len(train_images)
num_test_images = len(test_images)

def train_batch(batch_size):
    global train_batch_pointer
    images = []
    angles = []
    for i in range(batch_size):
        #将像素值归一化
        images.append(scipy.misc.imresize(scipy.misc.imread(train_images[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        angles.append([train_angles[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return images, angles

def test_batch(batch_size):
    global test_batch_pointer
    images = []
    angles = []
    for i in range(batch_size):
        #将像素值归一化
        images.append(scipy.misc.imresize(scipy.misc.imread(test_images[(test_batch_pointer + i) % num_test_images])[-150:], [66, 200]) / 255.0)
        angles.append([test_angles[(test_batch_pointer + i) % num_test_images]])
    test_batch_pointer += batch_size
    return images, angles


## 训练



#保存记录
saver = tf.train.Saver()
save_path = './checkpoints'
sess = tf.InteractiveSession()



train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

total_iterations = 0 #在整个训练集上训练的次数

def optimize(num_iterations): #训练次数由num_iterations决定
    start_time = time.time()
    global total_iterations
    writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
    t = total_iterations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(t, t + num_iterations):
        for j in range(int(num_images/batch_size)):
            image_batch, angle_batch = train_batch(batch_size)
            train.run(feed_dict={image: image_batch, angle: angle_batch, keep_prob: 0.8})
            if j % 10 == 0:
                image_batch_, angle_batch_ = test_batch(batch_size)
                loss_on_test = loss.eval(feed_dict={image:image_batch_, angle: angle_batch_, keep_prob: 1.0})
                print("total_iterations: %d, Step: %d, Loss_On_Test: %g" % (total_iterations , j, loss_on_test))
                checkpoint_path = os.path.join(save_path, "model.ckpt")
                saver.save(sess=sess, save_path=checkpoint_path)
        total_iterations += 1
    writer.close()
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))




optimize(1) #训练一次


# ## helper function:可视化卷积核（感谢[Hvass-Labs](https://github.com/Hvass-Labs)）


def plot_conv_weights(weights, input_channel=0):
    ## input_channel 是为了区分是哪一个通道的卷积核
    
    
    w = sess.run(weights) #得到卷积核的权重，这里包含了某一层的所有卷积核

    ## 用来归一化
    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3] #取到卷积核的个数，例如shape为[5, 5, 3, 24]，则对应有24个卷积核

    num_grids = math.ceil(math.sqrt(num_filters)) #以num_grids x num_grids的方式呈现图片
    
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        ax.set_xticks([]) #去掉多余的数轴
        ax.set_yticks([])
    
    plt.show()


# ## helper function:可视化卷积层（感谢[Hvass-Labs](https://github.com/Hvass-Labs)）


def plot_conv_layer(layer, image_for_visual):
    
    feed_dict = {image: [image_for_visual]} #计算图的输入值为image

    values = sess.run(layer, feed_dict=feed_dict) #实际得到的卷积层

    num_filters = values.shape[3] #获得卷积层的层数

    num_grids = math.ceil(math.sqrt(num_filters)) #呈现方式为 num_grids x num_grids
    
    fig, axes = plt.subplots(num_grids, num_grids)
    
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# ## 可视化卷积核



plot_conv_weights(W_conv1) #这一层共24个3通道卷积核，


# ## 可视化卷积层


image_for_visual = scipy.misc.imresize(scipy.misc.imread(train_images[1])[-150:], [66, 200]) / 255.0
plot_conv_layer(conv1, image_for_visual) #这一层共有24张输出

