#this file defines a new type of Keras layer named RoiPoolingConv. This layer is used in the Fast RCNN.
#basics of defining a new layer are stated here: https://keras.io/layers/writing-your-own-keras-layers/

from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used. Region of interest is the same as a region proposal
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        4D tensor with shape:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, pool_size, pool_size, channels)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        #set the input variables to self
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        #define the number of channels
        self.nb_channels = input_shape[0][3]
        #normally in this method you declare your weights, but since we don't have any learnable weights in this layer, this is not needed in our case.

    def compute_output_shape(self, input_shape):
        #needed since output shape is different than input shape
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        #x should be [input_img, rois]
        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        outputs = []

        #loop through all rois
        for roi_idx in range(self.num_rois):
            #get coordinates
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            #here bilinear interpolation is used. This is a bit different than the max pooling described in the paper
            #however, it reaches the same goal and is easier to implement.
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        #keras layer that concatenates the outputs along one axis.
        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
