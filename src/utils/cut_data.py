import pathlib
import random
import numpy
import pandas
import PIL.Image
import astropy.io.fits
import astropy.wcs
import astroquery.vizier
from . import get_catalog

def get_spitzer_df(path, fac, b=[-0.8, 0.8], R=[0.1, 10], seed=None, catalog='churchwell', gal='mw'):
    return SpitzerDf(path, fac, b, R, seed, catalog, gal)


class SpitzerDf(object):
    path = None
    files = None
    df = None

    def __init__(self, path, fac, b, R, seed, catalog, gal):
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

        if isinstance(catalog, str):

            if catalog == 'churchwell':
                self.df = get_catalog.churchwell_bubble()
                self.df = self.df.set_index('name')
                pass

            elif catalog == 'mwp_1st':
                self.df = get_catalog.mwp1st_bubble()
                self.df = self.df.set_index('name')
                pass

            elif catalog == 'mwp_2nd':
                self.df = get_catalog.mwp2nd_bubble()
                self.df = self.df.set_index('name')
                pass

            elif catalog == 'wise':
                self.df = get_catalog.wise_hii()
                self.df = self.df.set_index('name')

            else:
                print('no catalog')
                pass

        else:
            self.df = catalog
            self.df = self.df.set_index('name')
            pass

        if fac != 0:
#             self._add_random_df(fac, b, R, seed)
#             self._add_random_df2(fac, b, R, seed)
            self._add_random_df3(fac, b, R, seed)
            pass

        self.df_org = self.df.copy()

        if gal == 'mw':
            self._get_dir()
            pass

        if gal == 'lmc':
            file = list(self.path.glob('*'))[0]
            file = str(file).split('/')[-1]
            self.df.loc[:, 'directory'] = file
            pass

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
            l_range = 2.5
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

    def _add_random_df2(self, fac, b, R, seed):
        random.seed(seed)
        numpy.random.seed(seed)
        l = sorted([int(i[8:11]) for i in self.files])
        l.pop(l.index(294))

        over_358p5 = self.df.loc[:, 'l']>358.5
        for i in self.df[over_358p5].loc[:, 'l'].index:
            self.df.loc[i, 'l'] -= 360
            pass

        l_bub = self.df.loc[:, 'l'].tolist()
        b_bub = self.df.loc[:, 'b'].tolist()
        R_bub = self.df.loc[:, 'Rout'].tolist()

        hist = numpy.histogram(self.df.loc[:,'Rout'], bins=25)
        num, y_ = hist[0]*fac, hist[1]
        diff_ = y_[1:]-y_[:-1]

        R_nbub = []
        for i in range(len(num)):
            R_nbub_ = numpy.array(
                [numpy.random.rand()*diff_[i] + y_[i] for j in range(num[i])]
            )
            R_nbub.append(R_nbub_)
            pass

        R_nbub = numpy.concatenate(R_nbub)

        # Generate coordinates and size randomly within specified range
        name, glon_li, glat_li, size_li, i_n = [], [], [], [], 1
        for R_nbub_ in R_nbub:
            l_range = 2.5
            flag = True
            while flag:
                l_center = random.choice(l)
                l_fac = numpy.random.rand()
                b_fac = numpy.random.rand()
                i_l = round((l_range*l_fac) + l_center - (l_range/2), 3)
                i_b = round((b[1] - b[0])*b_fac + b[0], 3)
                i_R = round(R_nbub_, 2)
                # Select one that does not overlap with the bubble catalog
                distance = [(i_l - j_l)**2 + (i_b - j_b)**2 for j_l, j_b in zip(l_bub, b_bub)]
                # Allows up to 1/5 of ring size
                _min = [(i_R/60 + (j_R/60)/5)**2 for j_R in R_bub]
                if all([_d > _m for _d, _m in zip(distance, _min)]):
                    name.append('F{}'.format(i_n))
                    glon_li.append(i_l)
                    glat_li.append(i_b)
                    size_li.append(i_R)
                    i_n += 1
                    flag = False
                    pass

        nbub = pandas.DataFrame({'name': name, 'l': glon_li, 'b': glat_li, 'Rout': size_li})
        nbub = nbub.set_index('name')
        # add columns for label
        self.df = self.df.assign(label=1)
        nbub = nbub.assign(label=0)
        self.df = self.df.append(nbub)[self.df.columns.tolist()]
        self.df = self.df.loc[(self.df.loc[:, 'Rout']>R[0])&(self.df.loc[:, 'Rout']<R[1])]

        under_0p0 = self.df.loc[:, 'l']<0
        for i in self.df[under_0p0].loc[:, 'l'].index:
            self.df.loc[i, 'l'] += 360
            pass

        return

    def _add_random_df3(self, fac, b, R, seed):
        random.seed(seed)
        numpy.random.seed(seed)
        l = sorted([int(i[8:11]) for i in self.files])
        l.pop(l.index(294))

        over_358p5 = self.df.loc[:, 'l']>358.5
        for i in self.df[over_358p5].loc[:, 'l'].index:
            self.df.loc[i, 'l'] -= 360
            pass

        l_bub = numpy.array(self.df.loc[:, 'l'].tolist())
        b_bub = numpy.array(self.df.loc[:, 'b'].tolist())
        R_bub = numpy.array(self.df.loc[:, 'Rout'].tolist())

        hist = numpy.histogram(self.df.loc[:,'Rout'], bins=25)
        num, y_ = hist[0]*fac, hist[1]
        diff_ = y_[1:]-y_[:-1]

        R_nbub = []
        for i in range(len(num)):
            R_nbub_ = numpy.array(
                [numpy.random.rand()*diff_[i] + y_[i] for j in range(num[i])]
            )
            R_nbub.append(R_nbub_)
            pass

        R_nbub = numpy.concatenate(R_nbub)

        # Generate coordinates and size randomly within specified range
        name, glon_li, glat_li, size_li, i_n = [], [], [], [], 1
        for R_nbub_ in R_nbub:
            s_mask = (R_nbub_/R_bub>1/5)&(R_nbub_/R_bub<5)
            l_range = 2.5
            flag = True
            while flag:
                l_center = random.choice(l)
                l_fac = numpy.random.rand()
                b_fac = numpy.random.rand()
                i_l = round((l_range*l_fac) + l_center - (l_range/2), 3)
                i_b = round((b[1] - b[0])*b_fac + b[0], 3)
                i_R = round(R_nbub_, 2)
                # Select one that does not overlap with the bubble catalog
                distance = [(i_l - j_l)**2 + (i_b - j_b)**2 for j_l, j_b in zip(l_bub[s_mask], b_bub[s_mask])]
                # Allows up to 1/5 of ring size
                _min = [(i_R/60 + (j_R/60)/5)**2 for j_R in R_bub[s_mask]]
                if all([_d > _m for _d, _m in zip(distance, _min)]):
                    name.append('F{}'.format(i_n))
                    glon_li.append(i_l)
                    glat_li.append(i_b)
                    size_li.append(i_R)
                    i_n += 1
                    flag = False
                    pass

        nbub = pandas.DataFrame({'name': name, 'l': glon_li, 'b': glat_li, 'Rout': size_li})
        nbub = nbub.set_index('name')
        # add columns for label
        self.df = self.df.assign(label=1)
        nbub = nbub.assign(label=0)
        self.df = self.df.append(nbub)[self.df.columns.tolist()]
        self.df = self.df.loc[(self.df.loc[:, 'Rout']>R[0])&(self.df.loc[:, 'Rout']<R[1])]

        under_0p0 = self.df.loc[:, 'l']<0
        for i in self.df[under_0p0].loc[:, 'l'].index:
            self.df.loc[i, 'l'] += 360
            pass

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
        mask_min = self.df.loc[:, 'l']>l_min
        mask_max = self.df.loc[:, 'l']<l_max
        self.df = self.df.loc[mask_min&mask_max]
        pass

    def limit_b(self, b_min, b_max):
        mask_min = self.df.loc[:, 'b']>b_min
        mask_max = self.df.loc[:, 'b']<b_max
        self.df = self.df.loc[mask_min&mask_max]
        pass

    def limit_R(self, R_min, R_max):
        mask_min = self.df.loc[:, 'Rout']>R_min
        mask_max = self.df.loc[:, 'Rout']<R_max
        self.df = self.df.loc[mask_min&mask_max]
        return

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
        return sorted(dir_)


class CutTable(object):
    path = None
    df = None
    header = None
    data = None

    def __init__(self, path, df, margin):
        self.path = path
        self.df = df
        self.df = self.df.assign(margin=numpy.nan)
        self.df = self.df.assign(x_pix_min=0)
        self.df = self.df.assign(x_pix_max=0)
        self.df = self.df.assign(y_pix_min=0)
        self.df = self.df.assign(y_pix_max=0)

        if ('l' in self.df.columns.to_list()) & \
           ('b' in self.df.columns.to_list()):
            self.coord = ['l', 'b']
            pass

        elif ('ra' in self.df.columns.to_list()) & \
             ('dec' in self.df.columns.to_list()):
            self.coord = ['ra', 'dec']
            pass

        else:
            print('bad coordinates')
            pass

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
        info = self.df.reset_index()[self.df.index==obj].to_dict('records')[0]
        data = self.cut_img(obj)
        return data, info

    def calc_pix(self, obj, margin):
        series = self.df.loc[obj]
#         coord0_min = series[self.coord[0]] - margin*series['Rout']/60
#         coord1_min = series[self.coord[1]] - margin*series['Rout']/60
#         coord0_max = series[self.coord[0]] + margin*series['Rout']/60
#         coord1_max = series[self.coord[1]] + margin*series['Rout']/60
#         x_pix_min, y_pix_min = self.w.all_world2pix(coord0_max, coord1_min, 0)
#         x_pix_max, y_pix_max = self.w.all_world2pix(coord0_min, coord1_max, 0)
        coord0 = series[self.coord[0]]
        coord1 = series[self.coord[1]]
        x_pix_cen, y_pix_cen = self.w.all_world2pix(coord0, coord1, 0)
        x_pix_min = x_pix_cen - margin*series['Rout']*60/2
        x_pix_max = x_pix_cen + margin*series['Rout']*60/2
        y_pix_min = y_pix_cen - margin*series['Rout']*60/2
        y_pix_max = y_pix_cen + margin*series['Rout']*60/2

        ### 以下8行(コメントアウトを含む)は一時的なもの (all_world2pixのマニュアルを見ないといけない)
        if len(x_pix_min.shape) != 0: x_pix_min = x_pix_min[0]
        if len(y_pix_min.shape) != 0: y_pix_min = y_pix_min[0]
        if len(x_pix_max.shape) != 0: x_pix_max = x_pix_max[0]
        if len(y_pix_max.shape) != 0: y_pix_max = y_pix_max[0]
#         print(x_pix_min.shape)
#         print(y_pix_min.shape)
#         print(x_pix_max.shape)
#         print(y_pix_max.shape)

        r_pix = int(((x_pix_max - x_pix_min)/2 + (y_pix_max - y_pix_min)/2)/2)
        x_pix, y_pix = self.w.all_world2pix(series[self.coord[0]], series[self.coord[1]], 0)

        ### 以下4行(コメントアウトを含む)は一時的なもの (all_world2pixのマニュアルを見ないといけない)
        if len(x_pix.shape) != 0: x_pix = x_pix[0]
        if len(y_pix.shape) != 0: y_pix = y_pix[0]
#         print(x_pix.shape)
#         print(y_pix.shape)

        x_pix_min = max(0, int(numpy.round(x_pix)) - r_pix)
        x_pix_max = max(0, int(numpy.round(x_pix)) + r_pix)
        y_pix_min = max(0, int(numpy.round(y_pix)) - r_pix)
        y_pix_max = max(0, int(numpy.round(y_pix)) + r_pix)
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
        rgb = numpy.expand_dims(rgb, axis=0)
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
