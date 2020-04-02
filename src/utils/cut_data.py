import pathlib
import random
import numpy
import pandas
import PIL.Image
import astropy.io.fits
import astroquery.vizier
import tensorflow

def get_bubble_df():
    # make instance
    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    # load bub_2006
    bub_2006 = viz.query_constraints(catalog='J/ApJ/649/759/bubbles')[0].to_pandas()
    bub_2006.loc[:, '__CPA2006_'] = bub_2006.loc[:, '__CPA2006_'].str.decode('utf-8')
    bub_2006.loc[:, 'MFlags'] = bub_2006.loc[:, 'MFlags'].str.decode('utf-8')
    # load bub_2007
    bub_2007 = viz.query_constraints(catalog='J/ApJ/670/428/bubble')[0].to_pandas()
    bub_2007.loc[:, '__CWP2007_'] = bub_2007.loc[:, '__CWP2007_'].str.decode('utf-8')
    bub_2007.loc[:, 'MFlags'] = bub_2007.loc[:, 'MFlags'].str.decode('utf-8')
    # convert to pandas for 2006
    bub_2006.rename(columns={'__CPA2006_': 'name'}, inplace=True)
    bub_2006.rename(columns={'GLON': 'l'}, inplace=True)
    bub_2006.rename(columns={'GLAT': 'b'}, inplace=True)
    bub_2006.rename(columns={'__R_': '<R>'}, inplace=True)
    bub_2006.rename(columns={'__T_': '<T>'}, inplace=True)
    bub_2006.rename(columns={'MFlags': 'Flags'}, inplace=True)
    bub_2006.rename(columns={'_RA.icrs': 'RA.icrs'}, inplace=True)
    bub_2006.rename(columns={'_DE.icrs': 'DE.icrs'}, inplace=True)
    bub_2006 = bub_2006.set_index('name')
    # convert to pandas for 2007
    bub_2007.rename(columns={'__CWP2007_': 'name'}, inplace=True)
    bub_2007.rename(columns={'GLON': 'l'}, inplace=True)
    bub_2007.rename(columns={'GLAT': 'b'}, inplace=True)
    bub_2007.rename(columns={'__R_': '<R>'}, inplace=True)
    bub_2007.rename(columns={'__T_': '<T>'}, inplace=True)
    bub_2007.rename(columns={'MFlags': 'Flags'}, inplace=True)
    bub_2007.rename(columns={'_RA.icrs': 'RA.icrs'}, inplace=True)
    bub_2007.rename(columns={'_DE.icrs': 'DE.icrs'}, inplace=True)
    for i in bub_2007.index:
        bub_2007.loc[i, 'name'] = bub_2007.loc[i, 'name'].replace(' ', '')
        pass
    bub_2007 = bub_2007.set_index('name')
    # concat 2006 and 2007
    bub = pandas.concat([bub_2006, bub_2007])
    return bub

def get_spitzer_df(path, fac, b=[-0.8, 0.8], R=[0.1, 10], seed=None):
    return SpitzerDf(path, fac, b, R, seed)


class SpitzerDf(object):
    path = None
    files = None
    df = None
    
    def __init__(self, path, fac, b, R, seed):
        '''
        path: str or pathlib.Path
        fac: int or float
        b: [min, max] deg
        R: [min, max] arcmin
        seed: int or float
        '''
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path).expanduser()
            pass
        
        self.path = path
        self.files = [i.name for i in list(self.path.glob('*'))]
        self.df = get_bubble_df()
        self._add_random_df(fac, b, R, seed)
        self.df_org = self.df.copy()
        self._get_dir()
        pass
    
    def _add_random_df(self, fac, b, R, seed):
        random.seed(seed)
        numpy.random.seed(seed)
        l = sorted([int(i[8:11]) for i in self.files])
        l_bub = self.df.loc[:, 'l'].tolist()
        b_bub = self.df.loc[:, 'b'].tolist()
        R_bub = self.df.loc[:, 'Rout'].tolist()
        # Generate coordinates and size randomly within specified range    
        name, glon_li, glat_li, size_li, i_n = [], [], [], [], 1
        while len(glon_li) < len(self.df)*fac:
            l_range = 2
            l_center = random.choice(l)
            l_fac = numpy.random.rand()
            b_fac = numpy.random.rand()
            s_fac = numpy.random.rand()
            i_l = round((l_range*l_fac) + l_center - (l_range/2), 3)
            i_b = round((b[1] - b[0])*b_fac + b[0], 3)
            i_R = round((R[1] - R[0])*s_fac + R[0], 2)
            # Select one that does not overlap with the bubble catalog
            distance = [(i_l - j_l)**2 + (i_b - j_b)**2 for j_l, j_b in zip(l_bub, b_bub)]
            _min = [(i_R/60 + j_R/60)**2 for j_R in R_bub]
            if all([_d > _m for _d, _m in zip(distance, _min)]):
                name.append('F{}'.format(i_n))
                glon_li.append(i_l)
                glat_li.append(i_b)
                size_li.append(i_R)
                i_n += 1
                pass
        nbub = pandas.DataFrame({'name': name, 'l': glon_li, 'b': glat_li, 'Rout': size_li})
        nbub = nbub.set_index('name')
        # add columns for label
        self.df = self.df.assign(label=1)
        nbub = nbub.assign(label=0)
        self.df = self.df.append(nbub)[self.df.columns.tolist()]
        self.df = self.df.loc[(self.df.loc[:, 'Rout']>R[0])&(self.df.loc[:, 'Rout']<R[1])]
        return
    
    def _get_dir(self):
        files = self.path.glob('*')
        over_358p5 = self.df.loc[:, 'l']>358.5
        for i in self.df[over_358p5].loc[:, 'l'].index:
            self.df.loc[i, 'l'] -= 360
            pass
        for file in files:
            file = str(file).split('/')[-1]
            l_center = float(file[8:11])
            l_min = l_center - 1.5
            l_max = l_center + 1.5
            _slice = (self.df.loc[:, 'l']>=l_min)&(self.df.loc[:, 'l']<l_max)
            for i in self.df.loc[_slice].index:
                self.df.loc[i, 'directory'] = file
                pass
            pass
        under_0p0 = self.df.loc[:, 'l']<0
        for i in self.df[under_0p0].loc[:, 'l'].index:
            self.df.loc[i, 'l'] += 360
            pass
        # drop NaN file line
        self.df = self.df.dropna(subset=['directory'])
        return
    
    def get_cut_table(self, dir_, margin=3):
        df = self.df.loc[self.df.loc[:, 'directory']==dir_]
        return CutTable(self.path/dir_, df, margin)
    
    def limit_l(self, l_min, l_max):
        pass
    
    def limit_b(self, b_min, b_max):
        pass
    
    def limit_R(self, R_min, R_max):
        pass
    
    def drop_label(self, label):
        self.df = self.df[self.df.loc[:,'label']!=label]
        return
    
    def drop_obj(self, objs):
        self.df = self.df.drop(objs, axis=0)
        return
    
    def select_obj(self, objs):
        self.df = self.df.loc[objs]
        return
    
    def reset_df(self):
        self.df = self.df_org
        return
    
    def get_dir(self):
        dir_ = self.df.loc[:, 'directory'].unique().tolist()
        return dir_


class CutTable(object):
    path = None
    df = None
    header = None
    data = None
    
    def __init__(self, path, df, margin):
        self.path = path
        self.df = df.drop("directory", axis=1)
        self.df = self.df.assign(margin=numpy.nan)
        self.df = self.df.assign(x_pix_min=0)
        self.df = self.df.assign(x_pix_max=0)
        self.df = self.df.assign(y_pix_min=0)
        self.df = self.df.assign(y_pix_max=0)
        rgb = ['r.fits', 'g.fits', 'b.fits']
        hdus = [astropy.io.fits.open(path/i)[0] for i in rgb]
        self.header = {
            'r': hdus[0].header,
            'g': hdus[1].header,
            'b': hdus[2].header,
        }
        self.data = {
            'r': hdus[0].data,
            'g': hdus[1].data,
            'b': hdus[2].data,
        }
        self.w = astropy.wcs.WCS(self.header['g'])
        [self.calc_pix(i, margin) for i in self.get_obj()]
        pass
    
    def __repr__(self):
        return '<CutTable path={}>'.format(self.path)
    
    def __getitem__(self, obj):
        info = dict(self.df.loc[obj])
        info['name'] = obj
        info['path'] = str(self.path)
        data = self.cut_img(obj)
        return data, info
    
    def calc_pix(self, obj, margin):
        series = self.df.loc[obj]        
        l_min = series['l'] - margin*series['Rout']/60
        b_min = series['b'] - margin*series['Rout']/60
        l_max = series['l'] + margin*series['Rout']/60
        b_max = series['b'] + margin*series['Rout']/60
        x_pix_min, y_pix_min = self.w.all_world2pix(l_max, b_min, 0)
        x_pix_max, y_pix_max = self.w.all_world2pix(l_min, b_max, 0)
        R_pix = int(((x_pix_max - x_pix_min)/2 + (y_pix_max - y_pix_min)/2)/2)
        x_pix, y_pix = self.w.all_world2pix(series['l'], series['b'], 0)
        x_pix_min = max(0, int(numpy.round(x_pix)) - R_pix)
        x_pix_max = max(0, int(numpy.round(x_pix)) + R_pix)
        y_pix_min = max(0, int(numpy.round(y_pix)) - R_pix)
        y_pix_max = max(0, int(numpy.round(y_pix)) + R_pix)
        self.df.loc[obj, 'margin'] = margin
        self.df.loc[obj, 'x_pix_min'] = x_pix_min
        self.df.loc[obj, 'x_pix_max'] = x_pix_max
        self.df.loc[obj, 'y_pix_min'] = y_pix_min
        self.df.loc[obj, 'y_pix_max'] = y_pix_max
        return
    
    def cut_img(self, obj):
        x_pix_min = self.df.loc[obj, 'x_pix_min']
        x_pix_max = self.df.loc[obj, 'x_pix_max']
        y_pix_min = self.df.loc[obj, 'y_pix_min']
        y_pix_max = self.df.loc[obj, 'y_pix_max']
        r = self.data['r'][y_pix_min:y_pix_max, x_pix_min:x_pix_max]
        g = self.data['g'][y_pix_min:y_pix_max, x_pix_min:x_pix_max]
        b = self.data['b'][y_pix_min:y_pix_max, x_pix_min:x_pix_max]
        rgb = numpy.stack([r, g, b], 2)
        rgb = self._padding_obj(rgb, x_pix_min, y_pix_min)
        rgb = numpy.flipud(rgb)
        return rgb
    
    def _padding_obj(self, data, x_pix_min, y_pix_min):
        pad = data.shape[0] - data.shape[1]
        if pad > 0 and x_pix_min == 0:
            data = numpy.pad(data, [(0, 0),(pad, 0), (0, 0)])
            pass    
        if pad > 0 and x_pix_min != 0:
            data = numpy.pad(data, [(0, 0),(0, pad), (0, 0)])
            pass
        if pad < 0 and y_pix_min == 0:
            data = numpy.pad(data, [(abs(pad), 0),(0, 0), (0, 0)])
            pass    
        if pad < 0 and y_pix_min != 0:
            data = numpy.pad(data, [(0, abs(pad)),(0, 0), (0, 0)])
            pass    
        return data
    
    def get_obj(self):
        return self.df.index.to_list()
    