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

def calc_dist(circle_df, point_df, column_name):
    dist_li = []
    for circle, point in circle_df.loc[:, column_name].str.split(',').items():
        for p in point:
            x_pix = point_df.loc[p, 'x_pix']
            y_pix = point_df.loc[p, 'y_pix']
            x_cen = (circle_df.loc[circle, 'x_pix_max'] + circle_df.loc[circle, 'x_pix_min'])//2
            y_cen = (circle_df.loc[circle, 'y_pix_max'] + circle_df.loc[circle, 'y_pix_min'])//2
            dist = ((x_pix-x_cen)**2 + (y_pix-y_cen)**2)**(1/2)
            dist /= circle_df.loc[circle, 'R']*60/2
            dist_li.append(dist)
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

### 円形マスク生成関数
def make_dist_arr(s):
    '''
    s: pixcel
    return: dist_arr
    '''
    _ = [(i - s//2)/(s//2) for i in range(s)]
    y, x = np.meshgrid(_, _)
    dist = (y**2 + x**2)**(1/2)
    return dist

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
