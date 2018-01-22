"""
This file contains the region proposal network model and the Fast RCNN model, together forming the Faster RCNN model.
It contains the VGG16 model up to layer 13 for Keras. This is inspired by the paper of Simonyan and Zisserman[1] and has been implemented by many on Keras.
Moreover, it contains the convolutional layers of the RPN and the classsifier of rpn.
Lastly, it includes the Fast RCNN which uses the region proposals to get classes and bounding boxes.
This code is largely copied from https://github.com/yhenon/keras-frcnn but we have adjusted it to our needs. In particular, we have added comments to make the code more readable.
# Reference
- [1] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import TimeDistributed
from keras import backend as K
from frcnn.RoiPoolingConv import RoiPoolingConv


#this is the standard keras VGG16 weight file. Should be downloaded from https://github.com/fchollet/deep-learning-models/releases
#There is also a weight file without top, i.e. fully connected, layers. Seems to me that we should use that one plus it is is was smaller (50mb). Don't know why they use this one.
def get_weight_path():
    return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    """
    Calculate the output width and height of the VGG
    By design the output lengthof the VGG is 16 times smaller than the input length of the image (caused by the 4 pooling layers)
    """
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):
    """
    Construct the VGG up to conv layer 13
    """
    # Determine proper input shape
    # should have 3 channels (RGB) and width and height are variable
    input_shape = (None, None, 3)

    if input_tensor is None:
        #Input makes a keras tensor of it. Is needed when inputting to a model
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    #define the layers of the VGG.
    # Block 1
    #conv2D performs 2D convolutional. Standard layer for convolution on images.
    #for first layer 64 is number of kernels. So depth of output will be 64.
    #(3,3) is the kernel size
    #standard stride is 1
    #padding=same makes sure that the ouput width and height is the same as the input
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #(2,2) is the factor by which to downscale (width and height)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    return x


def rpn(base_layers, num_anchors):
    """
    Construct the unique convolutional and classification layers of the RPN
    """
    #base_layers is in this case the outcome from nn_base
    #kernel_initializer='normal' should mean initial weights from a zero-mean Guassian distribution with standard deviation 0.01 according to the Faster RCNN paper.
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    #these layers are basically fully connected layers but are easier programmed in a convolutional fashion
    #x_class output has shape num_anchorsxwidth_VGGoutxheight_VGGout.
    #activation is sigmoid so it gives a score for 0 and 1. This means that for every anchor at every sliding window it gives a probability of it being an object. In the original paper, they give a probability for being an object and being background, but since if it is not an object it is background this implementation uses one number to entail both.
    #according to the paper, kernel_initalizer='uniform' should do the same thing as 'normal' and when looking into the code it seems it does.
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    #activation=linear basically doesn't convert the outputs to a certain range, they are kept as they are.
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=4, trainable=False):
    """
    This is the Fast RCNN.
    It uses the proposal regions, input_rois, as input.
    It then applies the ROIpooling to make every proposal region the same size.
    Then it applies 2 fully connected layers with droupout and finally a softmax layer to predict the class.
    """
    #define the number of pooling regions. The output size of ROIPoolingConv will be pooling_regionsxpooling_regions.
    #Standard is 7 as suggested by the paper
    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    #convert the input regions such that they all have the same size
    #use the output feature map from the VGG as input plust he rois calculated by x_regr in rpn
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    #two fully connected layers.
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    #predict the probability for each class by using softmax
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    #predict the bounding box for every class
    #Since it is about coordinates we use a linear activation.
    # note: no regression target for background class, hence nb_classes-1
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
