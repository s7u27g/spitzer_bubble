import time
import tqdm
import numpy
import tensorflow
from ..utils.processing import *

def calc_yx_num(cut_shape, slide_pix, map_shape):
    y_num = (int(map_shape[0])-cut_shape[0])//slide_pix[0] + 1
    x_num = (int(map_shape[1])-cut_shape[1])//slide_pix[1] + 1
#     y_num = round((map_shape[0]-cut_shape[0])/slide_pix[0]) + 1
#     x_num = round((map_shape[1])-cut_shape[1]/slide_pix[1]) + 1
    return y_num, x_num

def get_indices(d_idx, y_num, x_num):
    '''
    just calculate indices
    return: index array
    '''
    y = numpy.arange(0, y_num*d_idx[0], d_idx[0])
    x = numpy.arange(0, x_num*d_idx[1], d_idx[1])
    y_idx = (y[:,None]*1 + x[None,:]*0).ravel()[:,None]
    x_idx = (y[:,None]*0 + x[None,:]*1).ravel()[:,None]
#    x_idx, y_idx = numpy.meshgrid(x, y)
    indices = numpy.concatenate([
        y_idx.ravel()[:,None],
        x_idx.ravel()[:,None],
    ], axis=1)
    return indices

def calc_resize_num(cut_shape, ch_num, dtype, mem_size):
    bytes_ = {'float32': 4, 'float64': 8}
    resize_num = int((mem_size*10**9)/(cut_shape[0]*cut_shape[1]*ch_num*bytes_[dtype]))
    return resize_num

def calc_loop_num(all_num, part_num):
    loop_num = all_num//part_num
    if (all_num)%part_num==0: pass
    else: loop_num += 1
    return loop_num

def clip_data_st(data, st_idx, cut_shape):
    clip_data = []
    for i in st_idx:
        d = data[
            i[0]:i[0]+cut_shape[0],
            i[1]:i[1]+cut_shape[1]
        ][None]
        clip_data.append(d)
        pass
    clip_data = numpy.concatenate(clip_data)
    return clip_data

def inference(model, data, inf_num):
    inf_loop = calc_loop_num(len(data), inf_num)
    prob = []
    for i in range(inf_loop):
        d = data[i*inf_num:(i+1)*inf_num]
        p = model(d).numpy()[:, 1].tolist()
        prob += p
        pass
    return prob.astype(numpy.float32)


def calc_prob3(model, data, cut_shape, sld_fac):
    '''
    model: keras model object
    data: arr that is shape must be (y, x, color) or (y, x)
    cut_shape: tuple or list
    '''
    input_shape = model.input_shape[1:3]
    if data.ndim == 2: data = data[:,:,None]
    else: pass

    if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
        y_size = input_shape[0]
        x_size = input_shape[1]
        pass
    else:
        y_size = cut_shape[0]
        x_size = cut_shape[1]
        pass

    slide_pix = (int(round(cut_shape[0]/sld_fac)), int(round(cut_shape[1]/sld_fac)))
    map_shape = data.shape[:2]
    ch_num = data.shape[2]
    dtype = data.dtype.name

    yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
    st_idx = get_indices(slide_pix, *yx_num)
    resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.2)
    resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)

    inf_num = 2048
    prob = []
    for i in tqdm.tqdm(range(resize_loop)):
        _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
        d = clip_data_st(data, _st_idx, cut_shape)
        d = tensorflow.convert_to_tensor(d)
        d = resize(d, input_shape)
        d = standardize(d)
        prob += inference(model, d, inf_num)
        pass

    info = {
        'y_num': yx_num[0], 'x_num': yx_num[1],
        'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
        'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
        'y_org': data.shape[0], 'x_org': data.shape[1],
    }
    prob = numpy.array(prob)

    return info, prob

def calc_prob_cross(models, data, cut_shape, sld_fac):
    '''
    models: list of keras model object
    data: arr that is shape must be (y, x, color) or (y, x)
    cut_shape: tuple or list
    '''
    input_shape = models[0].input_shape[1:3]
    if data.ndim == 2: data = data[:,:,None]
    else: pass

    if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
        y_size = input_shape[0]
        x_size = input_shape[1]
        pass
    else:
        y_size = cut_shape[0]
        x_size = cut_shape[1]
        pass

    slide_pix = (int(round(cut_shape[0]/sld_fac)), int(round(cut_shape[1]/sld_fac)))
    map_shape = data.shape[:2]
    ch_num = data.shape[2]
    dtype = data.dtype.name

    yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
    st_idx = get_indices(slide_pix, *yx_num)
    resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.2)
    resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)

    inf_num = 2048
    probs = [[] for i in range(len(models))]
    for i in tqdm.tqdm(range(resize_loop)):
        _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
        d = clip_data_st(data, _st_idx, cut_shape)
        d = tensorflow.convert_to_tensor(d)
        d = resize(d, input_shape)
        d = standardize(d)

        for prob, model in zip(probs, models):
            prob += inference(model, d, inf_num)
            pass
        pass

    info = {
        'y_num': yx_num[0], 'x_num': yx_num[1],
        'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
        'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
        'y_org': data.shape[0], 'x_org': data.shape[1],
    }

    return info, probs

# def calc_prob(model, data, cut_shape):
#     '''
#     model: keras model object
#     data: arr that is shape must be (y, x, color) or (y, x)
#     cut_shape: tuple or list
#     '''
#     input_shape = model.input_shape[1:3]
#     if data.ndim == 2: data = data[:,:,None]
#     else: pass

#     if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
#         y_size = input_shape[0]
#         x_size = input_shape[1]
#         pass
#     else:
#         y_size = cut_shape[0]
#         x_size = cut_shape[1]
#         pass

#     slide_pix = (cut_shape[0]//10, cut_shape[1]//10)
#     map_shape = data.shape[:2]
#     ch_num = data.shape[2]
#     dtype = data.dtype.name

#     yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
#     st_idx = get_indices(slide_pix, *yx_num)
#     resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.2)
#     resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)

#     inf_num = 2048
#     prob = []
#     for i in tqdm.tqdm(range(resize_loop)):
#         _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
#         d = clip_data_st(data, _st_idx, cut_shape)
#         d = tensorflow.convert_to_tensor(d)
#         #d = resize(d, (100, 100))
#         #d = remove_star(d, (3,3))
#         d = resize(d, input_shape)
#         d = standardize(d)
#         prob += inference(model, d, inf_num)
#         pass

#     info = {
#         'y_num': yx_num[0], 'x_num': yx_num[1],
#         'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
#         'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
#         'y_org': data.shape[0], 'x_org': data.shape[1],
#     }
#     prob = numpy.array(prob)

#     return info, prob



# def _calc_prob(model, data, cut_shape):
#     input_shape = model.input_shape[1:3]
#     if data.ndim == 2: data = data[:,:,None]
#     else: pass

#     if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
#         y_size = input_shape[0]
#         x_size = input_shape[1]
#         pass
#     else:
#         y_size = cut_shape[0]
#         x_size = cut_shape[1]
#         pass

#     slide_pix = (cut_shape[0]//10, cut_shape[1]//10)
#     map_shape = data.shape[:2]
#     ch_num = data.shape[2]
#     dtype = data.dtype.name

#     yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
#     st_idx = get_indices(slide_pix, *yx_num)
#     resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.8)
#     resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)
#     print(resize_loop)

#     inf_num = 2048
#     prob = []
#     for i in tqdm.tqdm(range(resize_loop)):
#         _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
#         d = clip_data_st(data, _st_idx, cut_shape)
#         d = resize(d, input_shape)
#         d = standardize(d)
#         prob += inference(model, d, inf_num)
#         pass

#     info = {
#         'y_num': yx_num[0], 'x_num': yx_num[1],
#         'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
#         'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
#         'y_org': data.shape[0], 'x_org': data.shape[1],
#     }
#     prob = numpy.array(prob)

#     return info, prob



# def _calc_prob(model, data, cut_shape):
#     input_shape = model.input_shape[1:3]
#     if data.ndim == 2: data = data[:,:,None]
#     else: pass

#     if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
#         y_size = input_shape[0]
#         x_size = input_shape[1]
#         pass
#     else:
#         y_size = cut_shape[0]
#         x_size = cut_shape[1]
#         pass

#     slide_pix = (cut_shape[0]//10, cut_shape[1]//10)
#     map_shape = data.shape[:2]
# #     if len(data.shape) == 2: ch_num = 1
# #     else: ch_num = data.shape[2]
#     ch_num = data.shape[2]
#     dtype = data.dtype.name

#     yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
#     st_idx = get_indices(slide_pix, *yx_num)
#     resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.8)
#     resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)
#     print(resize_loop)

#     inf_num = 2048
#     prob = []
#     t0 = 0
#     t1 = 0
#     t2 = 0
#     t3 = 0
#     for i in tqdm.tqdm(range(10)):
#         _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
#         t = time.time()
#         d = clip_data_st(data, _st_idx, cut_shape)
#         t0 += t-time.time()
#         t = time.time()
#         d = resize(d, input_shape)
#         t1 += t-time.time()
#         t = time.time()
#         d = standardize(d)
#         t2 += t-time.time()
#         t = time.time()
#         prob += inference(model, d, inf_num)
#         t3 += t-time.time()
#         pass

#     info = {
#         'y_num': yx_num[0], 'x_num': yx_num[1],
#         'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
#         'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
#         'y_org': data.shape[0], 'x_org': data.shape[1],
#     }
#     prob = numpy.array(prob)

#     print(t0, t1, t2, t3)
#     return info, prob

# def calc_prob2(model, data, cut_shape):
#     '''
#     model: keras model object
#     data: arr that is shape must be (y, x, color) or (y, x)
#     cut_shape: tuple or list
#     '''
#     input_shape = model.input_shape[1:3]
#     if data.ndim == 2: data = data[:,:,None]
#     else: pass
#
#     if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
#         y_size = input_shape[0]
#         x_size = input_shape[1]
#         pass
#     else:
#         y_size = cut_shape[0]
#         x_size = cut_shape[1]
#         pass
#
#     slide_pix = (int(round(cut_shape[0]/10)), int(round(cut_shape[1]/10)))
#     map_shape = data.shape[:2]
#     ch_num = data.shape[2]
#     dtype = data.dtype.name
#
#     yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
#     st_idx = get_indices(slide_pix, *yx_num)
#     resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.2)
#     resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)
#
#     inf_num = 2048
#     prob = []
#     for i in tqdm.tqdm(range(resize_loop)):
#         _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
#         d = clip_data_st(data, _st_idx, cut_shape)
#         d = tensorflow.convert_to_tensor(d)
#         d = resize(d, input_shape)
#         d = standardize(d)
#         prob += inference(model, d, inf_num)
#         pass
#
#     info = {
#         'y_num': yx_num[0], 'x_num': yx_num[1],
#         'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
#         'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
#         'y_org': data.shape[0], 'x_org': data.shape[1],
#     }
#     prob = numpy.array(prob)
#
#     return info, prob

# def calc_prob4(model, data, cut_shape, sld_fac):
#     '''
#     model: keras model object
#     data: arr that is shape must be (y, x, color) or (y, x)
#     cut_shape: tuple or list
#     '''
#     input_shape = model.input_shape[1:3]
#     if data.ndim == 2: data = data[:,:,None]
#     else: pass
#
#     if input_shape[0]*input_shape[1]>cut_shape[0]*cut_shape[1]:
#         y_size = input_shape[0]
#         x_size = input_shape[1]
#         pass
#     else:
#         y_size = cut_shape[0]
#         x_size = cut_shape[1]
#         pass
#
#     slide_pix = (int(round(cut_shape[0]/sld_fac)), int(round(cut_shape[1]/sld_fac)))
#     map_shape = data.shape[:2]
#     ch_num = data.shape[2]
#     dtype = data.dtype.name
#
#     yx_num = calc_yx_num(cut_shape, slide_pix, map_shape)
#     st_idx = get_indices(slide_pix, *yx_num)
#     resize_num = calc_resize_num((y_size, x_size), ch_num, dtype, 0.2)
#     resize_loop = calc_loop_num(yx_num[0]*yx_num[1], resize_num)
#
#     inf_num = 2048
#     prob = []
#     arr = []
#     for i in tqdm.tqdm(range(resize_loop)):
#         _st_idx = st_idx[i*resize_num:(i+1)*resize_num]
#         d = clip_data_st(data, _st_idx, cut_shape)
#         d = tensorflow.convert_to_tensor(d)
#         d = resize(d, input_shape)
#         d = standardize(d)
#         prob += inference(model, d, inf_num)
#         arr.append(d.numpy())
#         pass
#
#     info = {
#         'y_num': yx_num[0], 'x_num': yx_num[1],
#         'y_cut': cut_shape[0], 'x_cut': cut_shape[1],
#         'y_sld': slide_pix[0], 'x_sld': slide_pix[1],
#         'y_org': data.shape[0], 'x_org': data.shape[1],
#     }
#     prob = numpy.array(prob)
#     arr = numpy.concatenate(arr, axis=0)
#
#     return info, prob, arr
