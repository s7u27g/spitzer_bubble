import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.signal import fftconvolve

def calc_sigma(fwhm):
    sigma = fwhm/(2*(2*np.log(2))**(1/2))
    return sigma

def calc_fwhm(sigma):
    fwhm = 2*sigma*(2*np.log(2))**(1/2)
    return fwhm

def make_gauss_kernel(sigma):
    shape = (round(sigma)*8 + 1, round(sigma)*8 + 1)
    f = lambda x: np.exp(-(x**2)/(2*sigma**2))
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    k = f((x**2 + y**2)**(1/2))
    return k/np.sum(k)

def smoothing(arr, shape):
    '''
    arr: The shape must be (num, y, x, color)
    shape: list or tuple
    '''
    arr_ = []
    if arr.shape[1] > shape[0]:
        for color in range(arr.shape[3]):
            fwhm = arr.shape[1]/shape[0]*2 # target fwhm
            sigma = calc_sigma(fwhm) # target sigma
            k_sigma = (sigma**2 - calc_sigma(2)**2)**(1/2)
            k = make_gauss_kernel(k_sigma)[None,:,:,None]
            arr_.append(fftconvolve(arr[:,:,:,color][:,:,:,None], k, mode='same'))
            pass
        arr_ = np.concatenate(arr_, axis=3)
        pass

    else:
        arr_ = arr
        pass

    return arr_

def resize(tensor, size, method='bilinear'):
    '''
    tensor: The shape must be (num, y, x, color)
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
    tensor: The shape must be (num, y, x, color)
    fac: crop factor
    '''
    tensor = tf.image.central_crop(
        image=tensor,
        central_fraction=fac,
    )
    return tensor

def rotate(tensor, deg):
    '''
    tensor: The shape must be (num, y, x, color)
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
    tensor: The shape must be (num, y, x, color)
    '''
    tensor = tf.image.flip_left_right(
        image=tensor
    )
    return tensor

def standardize(tensor):
    '''
    tensor: The shape must be (num, y, x, color)
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
    tensor: The shape must be (num, y, x, color)
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_mean = tf.math.reduce_mean(tensor, axis=1, keepdims=True)
    t_std = tf.math.reduce_std(tensor, axis=1, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = tf.reshape(tensor, s)
    return tensor

def normalize(tensor):
    '''
    tensor: The shape must be (num, y, x, color)
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
    tensor: The shape must be (num, y, x, color)
    '''
    s = tensor.shape
    tensor = tf.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_min = tf.math.reduce_min(tensor, axis=1, keepdims=True)
    tensor = tensor - t_min
    t_max = tf.math.reduce_max(tensor, axis=1, keepdims=True)
    tensor = tensor/t_max
    tensor = tf.reshape(tensor, s)
    return tensor

def remove_star(tensor, kernel_size):
    '''
    tensor: The shape must be (num, y, x, color)
    kernel_size: The type must be list or tuple
    '''
    tensor = tfa.image.median_filter2d(
        image=tensor,
        filter_shape=kernel_size,
    )
    return tensor



# def smoothing(arr, shape):
#     '''
#     arr: The shape must be (num, y, x, color)
#     shape: list or tuple
#     '''
#     arr_ = []
#     for color in range(arr.shape[3]):
#         fwhm = arr.shape[1]/shape[0]*2
#         sigma = _calc_sigma(fwhm)
#         k = _make_gauss_kernel(sigma)[None,:,:,None]
#         arr_.append(fftconvolve(arr[:,:,:,color][:,:,:,None], k, mode='same'))
#         pass
#     arr_ = np.concatenate(arr_, axis=3)
#     return arr_

# def crop_random(tensor, fac):
#     '''
#     tensor: The shape must be (num, y, x, color)
#     fac: crop factor
#     '''
#     tensor = tf.image.central_crop(
#         image=tensor,
#         central_fraction=fac,
#     )
#     return tensor

# def standardize_np(arr):
#     '''
#     arr: The shape must be (num, y, x, color)
#     '''
#     f = lambda x: ((x-np.mean(x))/np.std(x))
#     arr = np.concatenate([
#         np.concatenate([
#             f(a[:,:,i][:,:,None]) for i in range(a.shape[2])
#         ], axis=2)[None] for a in arr
#     ], axis=0)
#     return arr

# def nan2max(arr):
#     '''
#     arr: The shape must be (num, y, x, color)
#     '''
#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[3]):
#             arr_ = arr[i,:,:,j]
#             arr_[arr_!=arr_] = np.nanmax(arr_)
#             pass
#         pass
#     return arr
