import pathlib
import numpy as np
import pandas as pd

def func1(df, w):
    '''
    ピクセル値をDataFrameに付与
    '''
    from astropy.coordinates import SkyCoord
    ra = df.loc[:, 'ra']
    dec = df.loc[:, 'dec']
    x_pix, y_pix = w.world_to_pixel(
        SkyCoord(ra, dec, unit=['deg','deg'], frame='fk5')
    )
    df.loc[:, 'x_pix'] = np.round(x_pix).astype('int32')
    df.loc[:, 'y_pix'] = np.round(y_pix).astype('int32')
    return df

def count_point_around_circle(circle_df, point_df, size_fac):
    
    radius = circle_df.R*60/2
    x_cen = (circle_df.x_pix_max + circle_df.x_pix_min)//2
    y_cen = (circle_df.y_pix_max + circle_df.y_pix_min)//2
    
    p_li = []
    p_num = []
    for x, y, r in zip(x_cen, y_cen, radius):
        x -= point_df.loc[:, 'x_pix']
        y -= point_df.loc[:, 'y_pix']
        mask = (x**2 + y**2)**(1/2)<r*size_fac
        p_li.append(','.join(point_df.index[mask].to_list()))
        p_num.append(np.sum(mask))
        pass
    
    return p_li, p_num

def calc_dist(circle_df, point_df, column_name, rm_obj=None):
    dist_li = []
    for circle, point in circle_df.loc[:, column_name].str.split(',').items():
        
        if rm_obj:
            _point = []
            for p in point:
                if p in rm_obj:pass
                else: _point.append(p)
            point = _point
            pass
            
        for p in point:
            x_pix = point_df.loc[p, 'x_pix']
            y_pix = point_df.loc[p, 'y_pix']
            x_cen = (circle_df.loc[circle, 'x_pix_max'] + circle_df.loc[circle, 'x_pix_min'])//2
            y_cen = (circle_df.loc[circle, 'y_pix_max'] + circle_df.loc[circle, 'y_pix_min'])//2
            dist = ((x_pix-x_cen)**2 + (y_pix-y_cen)**2)**(1/2)
            dist /= circle_df.loc[circle, 'R']*60/2
            R_pix = circle_df.loc[circle, 'x_pix_max'] - circle_df.loc[circle, 'x_pix_min']
            R_pix = (R_pix/circle_df.loc[circle, 'margin'])/2
            R_pix = R_pix*np.pi
            dist_li.append([dist, len(point), R_pix])
            pass
        pass
    return dist_li

def calc_sigma(fwhm):
    sigma = fwhm/(2*(2*np.log(2))**(1/2))
    return sigma

def make_gauss_kernel(sigma):
    shape = (round(sigma)*8 + 1, round(sigma)*8 + 1)
    f = lambda x: np.exp(-(x**2)/(2*sigma**2))
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    k = f((x**2 + y**2)**(1/2))
    return k/np.sum(k)

### 中心からの距離 array を作る関数
def make_dist_arr(s):
    '''
    s: pixcel
    return: dist_arr
    '''
    _ = [(i - s//2)/(s//2) for i in range(s)]
    y, x = np.meshgrid(_, _)
    dist = (y**2 + x**2)**(1/2)
    return dist

### 円形マスク生成関数
# def make_circle_mask(s):
#     '''
#     s: pixcel
#     return: bool np.array (dim2)
#     '''
#     _ = [(i - s//2)/(s//2) for i in range(s)]
#     y, x = np.meshgrid(_, _)
#     dist = y**2 + x**2
#     mask_cir = (dist<1)
#     return mask_cir

### 円形マスク生成関数 (サイズ自由)
def make_circle_mask(s, fac=1):
    '''
    s: pixcel
    return: bool np.array (dim2)
    '''
    _ = [(i - s//2)/(s//2) for i in range(s)]
    y, x = np.meshgrid(_, _)
    dist = y**2 + x**2
    mask_cir = (np.sqrt(dist)<fac)
    return mask_cir

### 文字列を hdm 形式に整形する関数
def convert_hdms(list_):
    new_coord = []
    for c in list_:
        if len(c) == 2:
            ms = c[-1].split('.')
            
            if len(ms) == 2:
                dms = [c[0], ms[0], str(60*float(ms[1])/10)]
                pass
            
            else:
                dms = [c[0], ms[0], '0.0']
                pass
            
            new_coord.append(':'.join(dms))
            pass
        
        else:
            new_coord.append(':'.join(c))
            pass
        
        pass
    
    return new_coord

def path_formatter(path):
    if isinstance(path, pathlib.Path):
        return path
    else:
        return pathlib.Path(path)
    
def make_regfile_cir(infos, file, coord='fk5'):
    region_info = 'fk5\n'
    
    for info in infos:
        cir = 'circle(' + \
              str(info['ra']) + ',' + \
              str(info['dec']) + ',' + \
              str(info['R']) + \
              '")\n'
        region_info += cir
        pass
    
    with open(file, 'w') as f:
        f.write(region_info)
        pass
    
    return

def make_regfile_dot(infos, file, coord='fk5'):
    region_info = 'fk5\n'
    
    for info in infos:
        point = 'point(' + \
              str(info['ra']) + ',' + \
              str(info['dec']) + \
              ') # point=x\n'
        region_info += point 
        pass
    
    with open(file, 'w') as f:
        f.write(region_info)
        pass
    
    pass

def _open_json(file):
    '''
    same func exist in src/utils/file_utils
    '''
    import simplejson as json
    with open(file, 'r') as f:
        infos = json.load(f)
        pass
    return infos

def json2reg(json_path, shape='circle', R_unit=None):
    import simplejson as json
    json_path = path_formatter(json_path)
    save_path = json_path.parent.parent/'reg'/(json_path.stem+'.reg')
    df = pd.DataFrame(_open_json(json_path))
    if shape=='circle':
        if R_unit=='deg': df['R'] *= 60
        elif R_unit=='arcmin': pass
        elif R_unit=='arcsec': df['R'] /= 60       
        else: pass
        infos = df.to_dict('records')
        make_regfile_cir(infos=infos, file=save_path, coord='fk5')
        pass
    elif shape=='point':
        make_regfile_dot(infos=infos, file=save_path, coord='fk5')
        infos = df.to_dict('records')
        pass
    pass
