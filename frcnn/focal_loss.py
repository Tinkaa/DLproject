import tensorflow as tf
from keras import backend as K


epsilon = 1e-4


# smooth l1
def rpn_smooth_l1_regr(num_anchors, sigma=3.0):
    sigma_2 = sigma ** 2
    

    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * sigma_2 * x * x) + (1 - x_bool) * (x_abs - 0.5 / sigma_2))) / K.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


# focal loss
def rpn_focal_loss_cls(num_anchors, alpha=0.25, gamma=2.0):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        y_true_1 = y_true[:, :, :, :num_anchors]
        y_true_2 = y_true[:, :, :, num_anchors:]

        # Compute focal weight
        alpha_factor = K.ones_like(y_true_2) * alpha
        alpha_factor = tf.where(K.equal(y_true_2, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(y_true_2, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        # Compute focal loss
        loss = focal_weight * K.binary_crossentropy(y_true_2, y_pred)
        loss = y_true_1 * loss

        return K.sum(loss) / K.sum(epsilon + y_true_1)
    
    return rpn_loss_cls_fixed_num