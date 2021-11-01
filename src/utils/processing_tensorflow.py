import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def array2tensor(arr):
    '''
    arr: The shape must be NHWC
    return: pytorch tensor (shape is NCHW)
    '''
    tensor = tf.convert_to_tensor(arr)
    return tensor

def tensor2array(tensor):
    '''
    tensor: The shape must be NCHW
    return: pytorch tensor (shape is NHWC)
    '''
    arr = tensor.numpy()
    return arr

def resize(tensor, size, method='bilinear'):
    '''
    tensor: The shape must be NHWC
    size: The type must be list or tuple
    '''
    tensor = tf.image.resize(
        images=tensor,
        size=size,
        method=method,
    )
    return tensor

def crop(tensor, fac):
    '''
    tensor: The shape must be NHWC
    fac: crop factor
    '''
    tensor = tf.image.central_crop(
        image=tensor,
        central_fraction=fac,
    )
    return tensor

def rotate(tensor, deg):
    '''
    tensor: The shape must be NHWC
    deg: rotate angle [deg]
    '''
    rad = deg*(np.pi/180)
    tensor = tfa.image.rotate(
        images=tensor,
        angles=rad,
        interpolation='BILINEAR',
    )
    return tensor

def refrect(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    tensor = tf.image.flip_left_right(
        image=tensor
    )
    return tensor

def standardize(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = tf.reshape(tensor, s)
    return tensor

def standardize_all(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = tf.reshape(tensor, s)
    return tensor

def standardize_3sig(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = tf.where(tensor>_max, _max, tensor)
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize_all(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize_3sig(tensor):
    '''
    tensor: The shape must be NHWC
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = tf.where(tensor>_max, _max, tensor)
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize_3sig_v2(tensor):
    '''
    tensor: the shape must be nhwc
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    _max = t_mean+(3*t_std)
    _min = t_mean-(3*t_std)
    tensor = tf.where(tensor>_max, _max, tensor)
    tensor = tf.where(tensor<_min, _min, tensor)
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize_3sig_v3(tensor):
    '''
    tensor: the shape must be nhwc
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2], s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    _max = t_mean+(3*np.sqrt(t_std))
    _min = t_mean-(3*np.sqrt(t_std))
    tensor = tf.where(tensor>_max, _max, tensor)
    tensor = tf.where(tensor<_min, _min, tensor)
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize_all_3sig(tensor):
    '''
    tensor: the shape must be nhwc
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = tf.where(tensor>_max, _max, tensor)
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def remove_star(tensor, kernel_size):
    '''
    tensor: the shape must be nhwc
    kernel_size: the type must be list or tuple
    '''
    tensor = tfa.image.median_filter2d(
        image=tensor,
        filter_shape=kernel_size,
    )
    return tensor

# def crop_random(tensor, fac):
#     '''
#     tensor: the shape must be (num, y, x, color)
#     fac: crop factor
#     '''
#     tensor = tf.image.central_crop(
#         image=tensor,
#         central_fraction=fac,
#     )
#     return tensor
