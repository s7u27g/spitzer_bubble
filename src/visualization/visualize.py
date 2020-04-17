import numpy as np
from PIL import Image, ImageDraw

def data_view(col, imgs, infos=None):
    '''
    col: number of columns
    imgs: tensor or nparray with a shape of (?, y, x, 1) or (?, y, x, 3)
    infos: dictonary from CutTable
    '''
    imgs = np.uint8(imgs[:, :, :, 0]) if imgs.shape[3] == 1 else np.uint8(imgs)
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