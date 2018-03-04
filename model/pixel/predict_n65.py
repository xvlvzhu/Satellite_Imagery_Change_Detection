# -*- coding: utf8 -*-


import numpy as np
import time

# 邻域数量
n = 65
size = n*n*8
# 单个批次的节点数量
batch_size = 1079#7553
positive_sample_batch_size = 600
negative_sample_batch_size = 900
model_path = "D:\\demo\\pycharmWorksapce\\testTensorflow\\model_10_31_n65\\model.ckpt"

index_p = 0
index_n = 0

print("Copy end.....")
quickbird2015 = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\quickbird2015_2.npy").swapaxes(0, 1).swapaxes(1, 2)
quickbird2017 = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\quickbird2017_2.npy").swapaxes(0, 1).swapaxes(1, 2)
print(quickbird2015.shape)
print(quickbird2017.shape)
image_dimension = quickbird2015.shape

print("Load end.....")

# 3. 扩展原始图像数据
extend = n - 1
distance = int(extend / 2)

quickbird2015_extend = np.zeros((image_dimension[0] + extend, image_dimension[1] + extend, image_dimension[2]))
quickbird2017_extend = np.zeros((image_dimension[0] + extend, image_dimension[1] + extend, image_dimension[2]))
quickbird2015_extend[distance: image_dimension[0] + distance, distance:image_dimension[1] + distance, :] = quickbird2015
quickbird2017_extend[distance: image_dimension[0] + distance, distance:image_dimension[1] + distance, :] = quickbird2017


# 获取中心点的邻域特征，邻域的数量由传入的参数指定，要求传入参数为奇数
# i:"横坐标", j:"纵坐标"
def get_feature(i, j):
    feature2015 = quickbird2015_extend[i:i + n, j:j + n, :]
    feature2017 = quickbird2017_extend[i:i + n, j:j + n, :]
    feature = np.concatenate((feature2015, feature2017), 2)

    # 将3维数组转为1维数组
    return feature.reshape(-1)


# 批量获取特征进行预测
def get_batch(i, j):
    count = 0
    batch_features = []
    begin_flag = False

    for row in range(i, image_dimension[0]):
        for column in range(image_dimension[1]):
            if not begin_flag:
                if column < j:
                    continue
                else:
                    begin_flag = True
            if begin_flag:
                count += 1
                batch_features.append(get_feature(row, column))

            # 结束标志
            end_flag = row == image_dimension[0] - 1 and column == image_dimension[1] - 1
            if count == batch_size or end_flag:

                if end_flag:
                    next_i = -1  # 下个批次开始的行号
                    next_j = -1  # 下个批次开始的列号
                else:
                    next_i = row + 1 if column + 1 == image_dimension[1] else row
                    next_j = 0 if column + 1 == image_dimension[1] else column + 1

                return {"end_flag": end_flag, "next_i": next_i, "next_j": next_j,
                        "batch_features": np.array(batch_features)}


import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, n*n*8])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 8, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,n, n, 8])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 96, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([1 * 1 * 128, 1024])
b_fc1 = bias_variable([1024])
h_pool4_flat = tf.reshape(h_pool4, [-1, 1 * 1 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
xxx = tf.argmax(y_conv, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

all_test_x = tf.placeholder(tf.float32, shape=[None, size])

saver = tf.train.Saver()

with tf.Session() as sess:


    saver.restore(sess, model_path)
    print("predict begin....")
    print(print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    out = []
    batch = get_batch(0, 0)
    while not batch['end_flag']:
        all_test_x_1 = batch['batch_features'] / 2774.0
        all_test_y_1 = sess.run(xxx, feed_dict={x: all_test_x_1, keep_prob: 1.0})
        out += list(all_test_y_1)
        next_i = batch["next_i"]
        next_j = batch["next_j"]
        batch = get_batch(next_i, next_j)
    all_test_x_1 = batch['batch_features'] / 2774.0
    all_test_y_1 = sess.run(xxx, feed_dict={x: all_test_x_1, keep_prob: 1.0})
    out += list(all_test_y_1)

    np.save('E:\\天池\\20171026_初赛第二阶段(new)\\out\\out_10_30_n65.npy', np.array(out,dtype=np.uint8).reshape((3000, 15106)))
    print("success.......")
    print(print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # tf.gfile.Copy('./out', os.path.join(FLAGS.buckets, 'out_10_30.npy'))
