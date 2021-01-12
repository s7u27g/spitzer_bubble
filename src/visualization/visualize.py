import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.patches
from astropy.units import arcmin, deg



def data_view(col, imgs, infos=None):
    '''
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    '''
    imgs = np.uint8(imgs[:,::-1,:,0]) if imgs.shape[3] == 1 else np.uint8(imgs[:,::-1])
    row = (lambda x, y: x//y if x/y-x//y==0.0 else x//y+1)(imgs.shape[0], col)
    dst = Image.new('RGB', (imgs.shape[1]*col, imgs.shape[2]*row))

    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        if infos != None:
            draw = ImageDraw.Draw(img)
            draw.text((3,3),infos[i]['name'])
        quo, rem = i//col, i%col
        dst.paste(img, (arr.shape[0]*rem, arr.shape[1]*quo))

    return dst

def data_view2(col, imgs, infos=None):
    '''
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    '''
    imgs = np.uint8(imgs[:,:,:,0]) if imgs.shape[3] == 1 else np.uint8(imgs[:,:])
    row = (lambda x, y: x//y if x/y-x//y==0.0 else x//y+1)(imgs.shape[0], col)
    dst = Image.new('RGB', (imgs.shape[1]*col, imgs.shape[2]*row))

    for i, arr in enumerate(imgs):
        img = Image.fromarray(arr)
        if infos != None:
            draw = ImageDraw.Draw(img)
            draw.text((3,3),infos[i]['name'])
        quo, rem = i//col, i%col
        dst.paste(img, (arr.shape[0]*rem, arr.shape[1]*quo))

    return ImageOps.flip(dst)

def _draw_circle(ax, info, linewitdh, edgecolor):
    ax.add_patch(
        matplotlib.patches.Circle(
            [info['l'], info['b']],
            radius=(info['Rout']*arcmin).to(deg).value,
            edgecolor = edgecolor,
            linewidth = linewitdh,
            linestyle = '-',
            facecolor = 'none',
            transform = ax.get_transform('galactic'),
        ),
    )
    pass

def draw_circles(ax, catalog, linewidth=1., edgecolor='w'):
    '''
    ax:
    catalog:
    '''
    [
        _draw_circle(ax, info, linewidth, edgecolor)
        for info in catalog.to_dict('records')
    ]
    pass

def _draw_point(ax, catalog, linewidth=1., edgecolor='w'):
    '''
    ax:
    catalog:
    '''
    [
        _draw_circle(ax, info, linewidth, edgecolor)
        for info in catalog.to_dict('records')
    ]
    pass

def draw_points(ax, catalog, linewidth=1., edgecolor='w'):
    '''
    ax:
    catalog:
    '''
    [
        _draw_circle(ax, info, linewidth, edgecolor)
        for info in catalog.to_dict('records')
    ]
    pass
