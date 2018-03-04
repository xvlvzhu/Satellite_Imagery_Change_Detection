from keras.models import Model
from keras.utils import layer_utils,to_categorical
from util.BilinearUpSampling import *
from util.resnet_helpers import *

# dimensions of images.
img_width, img_height = 256, 256

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 500
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
    x = Activation("softmax")(x)

    model = Model(img_input, x)
    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    # weights_path = os.path.expanduser("C:/Users/44369/.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    # print(weights_path)
    # model.load_weights(weights_path, by_name=True)
    return model

model = AtrousFCN_Resnet50_16s(input_shape=None, weight_decay=0., batch_momentum=0.9,
                               batch_shape=[batch_size, img_width, img_height, 8], classes=2)
model.compile(loss='categorical_crossentropy',#binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])

# transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
# datagen_train = SegmentationDataGenerator(transformer_train)
#
# transformer_val = RandomTransformer(horizontal_flip=False, vertical_flip=False)
# datagen_val = SegmentationDataGenerator(
#     transformer_val)

train_x = np.load("util/train_x_tif8.npy")/2580.0
train_y = to_categorical(np.load("util/train_y_tif8.npy")).reshape([-1,256,256,2])

model.fit(x = train_x,y=train_y,epochs=600,batch_size=batch_size,class_weight=[0.05,5])
model.save("train2.h5")
