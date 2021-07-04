import torch
import torchvision

methods = {
    'bilinear': torchvision.transforms.functional.InterpolationMode.BILINEAR,
    'bicubic': torchvision.transforms.functional.InterpolationMode.BICUBIC,
    'nearest': torchvision.transforms.functional.InterpolationMode.NEAREST,
}

def array2tensor(arr, device='cuda'):
    '''
    arr: The shape must be NHWC
    return: pytorch tensor (shape is NCHW)
    '''
    arr = arr.transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(arr)
    tensor = tensor.to(device)
    return tensor

def tensor2array(tensor):
    '''
    tensor: The shape must be NCHW
    return: pytorch tensor (shape is NHWC)
    '''
    tensor = tensor.to('cpu')
    arr = tensor.numpy()
    arr = arr.transpose(0, 2, 3, 1)
    return arr

def resize(tensor, size, method='bilinear'):
    '''
    tensor: The shape must be NCHW
    size: The type must be list or tuple
    '''
    tensor = torchvision.transforms.functional.resize(
        img=tensor,
        size=size,
        interpolation=methods[method],
    )
    return tensor

def crop(tensor, fac):
    '''
    tensor: The shape must be NCHW
    fac: crop factor
    '''
    if len(tensor.shape) == 4: s = (tensor.shape[2], tensor.shape[3])
    else: s = (tensor.shape[1], tensor.shape[2])
    tensor = torchvision.transforms.functional.center_crop(
        img=tensor,
        output_size=(round(s[0]*fac), round(s[1]*fac))
    )
    return tensor


def rotate(tensor, angle, method='bilinear'):
    '''
    tensor: The shape must be NCHW
    deg: rotate angle [deg]
    '''
    tensor = torchvision.transforms.functional.rotate(
        img=tensor,
        angle=angle,
        interpolation=methods[method],
        # expand=True,
    )
    return tensor

def reflect(tensor, vh='h'):
    '''
    tensor: The shape must be NCHW
    vh: str v or str h
    '''
    if vh=='h': tensor = torchvision.transforms.functional.hflip(tensor)
    elif vh=='v': tensor = torchvision.transforms.functional.vflip(tensor)
    else: pass
    return tensor

def standardize(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1], s[2]*s[3]])
    t_mean = torch.mean(tensor, dim=2, keepdims=True)
    t_std = torch.std(tensor, dim=2, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = torch.reshape(tensor, s)
    return tensor

def standardize_all(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_mean = torch.mean(tensor, dim=1, keepdims=True)
    t_std = torch.std(tensor, dim=1, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = torch.reshape(tensor, s)
    return tensor

def standardize_3sig(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1], s[2]*s[3]])
    t_mean = torch.mean(tensor, dim=2, keepdims=True)
    t_std = torch.std(tensor, dim=2, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = torch.where(tensor>_max, _max, tensor)
    t_mean = torch.mean(tensor, dim=2, keepdims=True)
    t_std = torch.std(tensor, dim=2, keepdims=True)
    tensor = (tensor-t_mean)/t_std
    tensor = torch.reshape(tensor, s)
    return tensor

def normalize(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1], s[2]*s[3]])
    t_min = torch.min(tensor, dim=2, keepdims=True).values
    tensor = tensor - t_min
    t_max = torch.max(tensor, dim=2, keepdims=True).values
    tensor = tensor/t_max
    tensor = torch.reshape(tensor, s)
    return tensor

def normalize_all(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_min = torch.min(tensor, dim=1, keepdims=True).values
    tensor = tensor - t_min
    t_max = torch.max(tensor, dim=1, keepdims=True).values
    tensor = tensor/t_max
    tensor = torch.reshape(tensor, s)
    return tensor

def normalize_3sig(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1], s[2]*s[3]])
    t_mean = torch.mean(tensor, dim=2, keepdims=True)
    t_std = torch.std(tensor, dim=2, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = torch.where(tensor>_max, _max, tensor)
    t_min = torch.min(tensor, dim=2, keepdims=True).values
    tensor = tensor - t_min
    t_max = torch.max(tensor, dim=2, keepdims=True).values
    tensor = tensor/t_max
    tensor = torch.reshape(tensor, s)
    return tensor

def normalize_all_3sig(tensor):
    '''
    tensor: The shape must be NCHW
    '''
    s = tensor.shape
    tensor = torch.reshape(tensor, [s[0], s[1]*s[2]*s[3]])
    t_mean = torch.mean(tensor, dim=1, keepdims=True)
    t_std = torch.std(tensor, dim=1, keepdims=True)
    _max = t_mean+(3*t_std)
    tensor = torch.where(tensor>_max, _max, tensor)
    t_min = torch.min(tensor, dim=1, keepdims=True).values
    tensor = tensor - t_min
    t_max = torch.max(tensor, dim=1, keepdims=True).values
    tensor = tensor/t_max
    tensor = torch.reshape(tensor, s)
    return tensor
