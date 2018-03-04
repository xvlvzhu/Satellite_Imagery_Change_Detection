
from keras.models import Model,load_model
from keras.utils import layer_utils,to_categorical
from util.BilinearUpSampling import *
from util.resnet_helpers import *
import matplotlib.pyplot as plt

img_width, img_height = 256, 256
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# data format
if K.image_data_format() == 'channels_first':
    input_shape = (8, img_width, img_height)
else:
    input_shape = (img_width, img_height, 8)


def AtrousFCN_Resnet50_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1),
                   batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(
        x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(
        x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay,
                       batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2),
                          batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2),
                              batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2),
                              batch_momentum=batch_momentum)(x)
    # classifying layer
    # x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1),
               kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    # x = Activation("sigmoid")(x)

    model = Model(img_input, x)
    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    # weights_path = os.path.expanduser("C:/Users/44369/.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    # print(weights_path)
    # model.load_weights(weights_path, by_name=True)
    return model


# model = Sequential()
# model.add(Conv2D(32, (3, 3),strides=(2,2), input_shape=input_shape,name='conv0'))
# model.add(BatchNormalization(axis=3,name='bn0'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(32, (3, 3),strides=(1,1),name='conv1'))
# model.add(BatchNormalization(axis=3,name='bn1'))
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
model = AtrousFCN_Resnet50_16s(input_shape=[img_width, img_height, 8], weight_decay=0., batch_momentum=0.9,
                               batch_shape=None, classes=2)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
# datagen_train = SegmentationDataGenerator(transformer_train)
#
# transformer_val = RandomTransformer(horizontal_flip=False, vertical_flip=False)
# datagen_val = SegmentationDataGenerator(
#     transformer_val)
model.load_weights("train2.h5")
train_x = np.load("util/train_x_tif8.npy")/2580.0
val_data = train_x[4:20]
print(val_data.shape)
val_y = to_categorical(np.load("util/train_y_tif8.npy")).reshape([-1,256,256,2])[4:20]

print(model.evaluate(val_data,val_y))
a = model.predict(val_data)
print(np.max(a))
a[a>=0.5]=1
a[a<0.5]=0
print(np.sum(a))
# print(a[15].reshape([256,256]))

plt.imshow(a[15,:,:,1].reshape([256,256]))
plt.imshow(val_y[15,:,:,1].reshape([256,256]))
plt.show()



