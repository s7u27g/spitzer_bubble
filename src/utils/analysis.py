import numpy
import skimage.transform
import scipy.ndimage
from scipy.signal import fftconvolve
from .processing import smoothing, resize


def make_prob_map(result_path, prob_info):
    prob = numpy.load(result_path/prob_info['file'])
    prob = prob.reshape((prob_info['y_num'], prob_info['x_num']))
    prob = prob.repeat(prob_info['y_sld'], axis=0)
    prob = prob.repeat(prob_info['x_sld'], axis=1)

    y_err = (prob_info['y_org'] - prob_info['y_cut'])%prob_info['y_sld']//2
    x_err = (prob_info['x_org'] - prob_info['x_cut'])%prob_info['x_sld']//2

    pad_yu = (prob_info['y_org']-prob.shape[0])//2 - y_err
    pad_yl = (prob_info['y_org']-prob.shape[0])//2 \
           + (prob_info['y_org']-prob.shape[0])%2 + y_err
    pad_xu = (prob_info['x_org']-prob.shape[1])//2 - x_err
    pad_xl = (prob_info['x_org']-prob.shape[1])//2 \
           + (prob_info['x_org']-prob.shape[1])%2 + x_err
    prob_map = numpy.pad(prob, [(pad_yu, pad_yl), (pad_xu, pad_xl)], 'constant')

    return prob_map

def make_prob_map2(result_path, prob_info, compression=1):
    resize_pix = [prob_info['y_org']//compression, prob_info['x_org']//compression]
    prob = numpy.load(result_path/prob_info['file'])
    prob = prob.reshape((prob_info['y_num'], prob_info['x_num']))
    prob = prob.repeat(prob_info['y_sld'], axis=0)
    prob = prob.repeat(prob_info['x_sld'], axis=1)

    y_err = (prob_info['y_org'] - prob_info['y_cut'])%prob_info['y_sld']//2
    x_err = (prob_info['x_org'] - prob_info['x_cut'])%prob_info['x_sld']//2

    pad_yu = (prob_info['y_org']-prob.shape[0])//2 - y_err
    pad_yl = (prob_info['y_org']-prob.shape[0])//2 \
           + (prob_info['y_org']-prob.shape[0])%2 + y_err
    pad_xu = (prob_info['x_org']-prob.shape[1])//2 - x_err
    pad_xl = (prob_info['x_org']-prob.shape[1])//2 \
           + (prob_info['x_org']-prob.shape[1])%2 + x_err
    prob_map = numpy.pad(prob, [(pad_yu, pad_yl), (pad_xu, pad_xl)], 'constant')
    prob_map = smoothing(prob_map[None,:,:,None], (resize_pix[0], resize_pix[1]))

    if compression==1:
        prob_map = prob_map[0,:,:,0]
        pass

    else:
        prob_map = resize(prob_map, (resize_pix[0], resize_pix[1]))[0,:,:,0]
        pass

    return prob_map

def make_prob_map3(result_path, prob_info, prob_th=0., pix_th=1, compression=1):
    resize_pix = [prob_info['y_org']//compression, prob_info['x_org']//compression]
    prob = numpy.load(result_path/prob_info['file'])
    prob = prob.reshape((prob_info['y_num'], prob_info['x_num']))
    prob = mask_func(prob, prob_th, pix_th)
    prob = prob.repeat(prob_info['y_sld'], axis=0)
    prob = prob.repeat(prob_info['x_sld'], axis=1)

    y_err = (prob_info['y_org'] - prob_info['y_cut'])%prob_info['y_sld']//2
    x_err = (prob_info['x_org'] - prob_info['x_cut'])%prob_info['x_sld']//2

    pad_yu = (prob_info['y_org']-prob.shape[0])//2 - y_err
    pad_yl = (prob_info['y_org']-prob.shape[0])//2 \
           + (prob_info['y_org']-prob.shape[0])%2 + y_err
    pad_xu = (prob_info['x_org']-prob.shape[1])//2 - x_err
    pad_xl = (prob_info['x_org']-prob.shape[1])//2 \
           + (prob_info['x_org']-prob.shape[1])%2 + x_err
    prob_map = numpy.pad(prob, [(pad_yu, pad_yl), (pad_xu, pad_xl)], 'constant')
    prob_map = smoothing(prob_map[None,:,:,None], (resize_pix[0], resize_pix[1]))

    if compression==1:
        prob_map = prob_map[0,:,:,0]
        pass

    else:
        prob_map = resize(prob_map, (resize_pix[0], resize_pix[1]))[0,:,:,0]
        pass

    return prob_map


def mask_func(prob_map_, prob_th, pix_th):
    prob_map = prob_map_.copy()
    mask = prob_map>prob_th
    prob_map *= mask
    ring_map, class_num = scipy.ndimage.label(prob_map)
    data_areas = scipy.ndimage.sum(prob_map, ring_map, numpy.arange(class_num+1))
    minsize = pix_th
    small_size_mask = data_areas < minsize
    small_mask = small_size_mask[ring_map.ravel()].reshape(ring_map.shape)
    num_masked_pixels = numpy.sum(small_mask)
    prob_map[small_mask] = 0
    return prob_map


def calc_gcen(arr):
    arr_nor = standardize(arr[None,:,:,None])[0,:,:,0]
    arr_nor = numpy.where(arr_nor<0, 0, arr_nor)
    gcen = scipy.ndimage.measurements.center_of_mass(arr_nor)
    return gcen
