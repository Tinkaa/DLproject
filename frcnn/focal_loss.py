'''
These loss functions are meant to replace the normal
loss functions for RPN.
We took inspiration from Focal Loss paper <https://arxiv.org/abs/1708.02002>
as we see similarity between RetinaNet and RPN, where the single
class we want to detect is foreground.
'''
import tensorflow as tf
from keras import backend as K

EPSILON = 1e-4


def rpn_smooth_l1_regr(num_anchors, sigma=3.0):
    '''
    Create and return loss regression function for RPN
    '''
    sigma_2 = sigma ** 2

    def rpn_loss_regr_fixed_num(y_true, y_pred):
        '''
        Smooth L1 loss
        '''
        _x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        # Absolute differences
        x_abs = K.abs(_x)
        # Piecewise indices for loss values
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        # Smooth l1 loss, x-> piecewise(|x| < 1, 0.5x^2, |x| - 0.5)
        # With normalizer by anchor states
        return K.sum(
            y_true[:, :, :, :4 * num_anchors] *
            (x_bool * (0.5 * sigma_2 * _x * _x) + (1 - x_bool) * (x_abs - 0.5 / sigma_2))
            ) / K.sum(EPSILON + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_focal_loss_cls(num_anchors, alpha=0.25, gamma=2.0):
    '''
    Create and return focal loss function for RPN
    alpha:  balance factor, 0 < alpha < 1,
            ratio of importance between foreground and background
    gamma:  focusing parameter (exponential factor)
            control the effect of modulating factor, which is 1 - p_t
            where p_1 = is the probability of being foreground
    The modulating factor increases the penalty for background,
    this suppressing them.
    '''
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        '''
        Focal loss function
        '''
        # Anchor states
        y_true_1 = y_true[:, :, :, :num_anchors]
        # Ground truth
        y_true_2 = y_true[:, :, :, num_anchors:]

        # Compute focal weight
        alpha_factor = K.ones_like(y_true_2) * alpha
        # Alpha multiplier for foreground, and (1-alpha) multiplier for background
        alpha_factor = tf.where(K.equal(y_true_2, 1), alpha_factor, 1 - alpha_factor)
        # negative focal weight
        focal_weight = tf.where(K.equal(y_true_2, 1), 1 - y_pred, y_pred)
        # FL = -alpha (1-p_t)^gamma
        focal_weight = alpha_factor * focal_weight ** gamma

        # Compute focal loss
        # Multiply with cross-entropy loss
        loss = focal_weight * K.binary_crossentropy(y_true_2, y_pred)
        loss = y_true_1 * loss

        # Normalize loss with anchor states
        return K.sum(loss) / K.sum(EPSILON + y_true_1)

    return rpn_loss_cls_fixed_num
