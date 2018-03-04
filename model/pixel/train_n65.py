# -*- coding: utf8 -*-

import sys
import argparse
import os
import numpy as np
import tensorflow as tf
import time

# 邻域数量
n = 65
size = n*n*8
# 单个批次的节点数量
batch_size = 7553
positive_sample_batch_size = 600
negative_sample_batch_size = 900

index_p = 0
index_n = 0

quickbird2015 = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\quickbird2015.npy").swapaxes(0, 1).swapaxes(1, 2)
quickbird2017 = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\quickbird2017.npy").swapaxes(0, 1).swapaxes(1, 2)
print(quickbird2015.shape)
print(quickbird2017.shape)
image_dimension = quickbird2015.shape


# 2. 读标注样本坐标
positive_c = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\label\\positive_c.npy")
np.random.shuffle(positive_c)
negative_c = np.load("E:\\天池\\20171026_初赛第二阶段(new)\\data\\label\\negative_c.npy")
np.random.shuffle(negative_c)
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


# 批次提取正样本
def get_positive_sample_batch():
    sample = []

    count = 0
    global index_p

    while count < positive_sample_batch_size:
        i = positive_c[index_p, 0]
        j = positive_c[index_p, 1]
        sample.append(get_feature(i, j))
        index_p += 1
        count += 1

        if index_p == 200000:
            index_p = 0

    return sample


# 批次提取负样本
def get_negative_sample_batch():
    sample = []

    count = 0
    global index_n

    while count < negative_sample_batch_size:
        i = negative_c[index_n, 0]
        j = negative_c[index_n, 1]
        sample.append(get_feature(i, j))
        index_n += 1
        count += 1

        if index_n == 300000:
            index_n = 0

    return sample


# 批次提取正负样本
def get_sample_batch():
    p = get_positive_sample_batch()
    n = get_negative_sample_batch()
    # 正负样本合并
    train_x = np.array(p + n) / 2774.0
    # 类标合并
    train_y_p = np.zeros((positive_sample_batch_size, 2))
    train_y_p[:, 1] = 1
    train_y_n = np.zeros((negative_sample_batch_size, 2))
    train_y_n[:, 0] = 1
    train_y = np.concatenate([train_y_p, train_y_n])

    train = np.hstack([train_x, train_y])

    return train

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, size])
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

sess.run(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get batch....")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in range(10000):
        sample_batch = get_sample_batch()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: sample_batch[:, 0:size], y_: sample_batch[:, size:size+2], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: sample_batch[:, 0:size], y_: sample_batch[:, size:size+2], keep_prob: 0.5})

    print("train begin....")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_accuracy_sum = 0.0
    for i in range(1000):
        sample_batch = get_sample_batch()
        train_accuracy = accuracy.eval(feed_dict={x: sample_batch[:, 0:size], y_: sample_batch[:, size:size+2], keep_prob: 1.0})
        train_accuracy_sum += train_accuracy
    print(train_accuracy_sum / 1000.0)

    saver.save(sess, "D:\\demo\\pycharmWorksapce\\testTensorflow\\model_10_31_n65\\model.ckpt")
    test_accuracy_sum = 0.0
    for i in range(1000):
        sample_batch = get_sample_batch()
        test_accuracy = accuracy.eval(feed_dict={x: sample_batch[:, 0:size], y_: sample_batch[:, size:size+2], keep_prob: 1.0})
        test_accuracy_sum += test_accuracy
    print(test_accuracy_sum / 1000.0)

    # print("predict begin....")
    # print(print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    # out = []
    # batch = get_batch(0, 0)
    # while not batch['end_flag']:
    #     all_test_x_1 = batch['batch_features'] / 2774.0
    #     all_test_y_1 = sess.run(xxx, feed_dict={x: all_test_x_1, keep_prob: 1.0})
    #     out += list(all_test_y_1)
    #     next_i = batch["next_i"]
    #     next_j = batch["next_j"]
    #     batch = get_batch(next_i, next_j)
    # all_test_x_1 = batch['batch_features'] / 2774.0
    # all_test_y_1 = sess.run(xxx, feed_dict={x: all_test_x_1, keep_prob: 1.0})
    # out += list(all_test_y_1)
    #
    # np.save('E:\\天池\\20171026_初赛第二阶段(new)\\out\\out_10_31.npy', np.array(out,dtype=np.uint8).reshape((3000, 15106)))
    print("success.......")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
