import tensorflow as tf
import numpy as np
imagecontent = tf.read_file("E:/tianchi/preliminary_1/originalLabel_256/2015_0_2_256_.jpg")
img = tf.image.decode_jpeg(imagecontent, channels=1)
imagecontent1 = tf.read_file("E:/tianchi/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClassAug/2007_000033.png")
img1 = tf.image.decode_jpeg(imagecontent1, channels=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(img)
    b = sess.run(img1)
    a1=np.array(a)
    b1=np.array(b)
    print(np.array(a).shape)
    print(np.array(b).shape)
    print(np.sum(b1))
    print(np.sum(a1))

