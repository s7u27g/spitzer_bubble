import numpy as np
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
