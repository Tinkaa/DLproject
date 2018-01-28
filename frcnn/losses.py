'''
Loss functions for RPN and Classifer net
- RPN has a loss function for regression and another one for classification
- Classifier net has a loss function for regression and another one for classification
'''
from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

LAMBDA_RPN_REGR = 1.0
LAMBDA_RPN_CLASS = 1.0

LAMBDA_CLS_REGR = 1.0
LAMBDA_CLS_CLASS = 1.0

EPSILON = 1e-4


def rpn_loss_regr(num_anchors):
    '''
    Create and return loss regression function for RPN
    '''
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        '''
        Smooth L1 loss
        L1 is less sensitive to outliers,
        which is the majority of pixels in an image
        '''
        _x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        # Anchor absolute differences
        x_abs = K.abs(_x)
        # Indices for piecewise loss function
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        # Smooth l1 loss, x-> piecewise(|x| < 1, 0.5x^2, |x| - 0.5)
        # With normalizer by anchor states
        return LAMBDA_RPN_REGR * K.sum(
            y_true[:, :, :, :4 * num_anchors] *
            (x_bool * (0.5 * _x * _x) + (1 - x_bool) * (x_abs - 0.5))
            ) / K.sum(EPSILON + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    '''
    Create and return loss classification function for RPN
    '''
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        '''
        Binary cross-entropy
        between ground truth and predicted class probability
        '''
        return LAMBDA_RPN_CLASS * K.sum(
            y_true[:, :, :, :num_anchors] * K.binary_crossentropy(
                y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])
            ) / K.sum(EPSILON + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    '''
    Create and return loss regression function for Classifier
    '''
    def class_loss_regr_fixed_num(y_true, y_pred):
        '''
        Smooth L1 loss
        '''
        _x = y_true[:, :, 4 * num_classes:] - y_pred
        # Absolute differences
        x_abs = K.abs(_x)
        # Piecewise loss indices
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

        # Smooth l1 loss, x-> piecewise(|x| < 1, 0.5x^2, |x| - 0.5)
        # With normalizer
        return LAMBDA_CLS_REGR * K.sum(
            y_true[:, :, :4 * num_classes] *
            (x_bool * (0.5 * _x * _x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                EPSILON + y_true[:, :, :4 * num_classes])

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    '''
    Classification loss function for Classifier
    Purely cross-entropy loss
    '''
    return LAMBDA_CLS_CLASS * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
