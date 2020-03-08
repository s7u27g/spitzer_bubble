import pathlib
import urllib.request
import numpy
import pandas
import random
import astropy.io.fits
import astropy.wcs
import astroquery.vizier
import PIL
import n2

def download_mipsgal_data(l, b, save_path):
    '''
    l must be int from 0 to 359.
    b must be str, 'p' or 'n'.
    '''
    save_path = pathlib.Path(save_path).expanduser()
    url = 'https://irsa.ipac.caltech.edu/data/SPITZER/MIPSGAL/images/mosaics24/'
    l = str(l)+'0'
    while len(l)!=4:
        l = '0'+l
        
    file_name = 'MG'+l+b+'005_024.fits'
    with urllib.request.urlopen(url+file_name) as u:
        with open(save_path/file_name, 'bw') as o:
            o.write(u.read())
            
    return

def download_glimpse_data(l, band, save_path):
    '''
    l must be int (recommended multiple of 3) from 0 to 359.
    band must be str, irac1, 2, 3, or 4.
    '''
    save_path = pathlib.Path(save_path).expanduser()
    band_dict = {'irac1': 'I1', 'irac2': 'I2', 'irac3': 'I3', 'irac4': 'I4'}
    url = 'https://irsa.ipac.caltech.edu//data/SPITZER/GLIMPSE/images/'
    if (-1<=l and l<11) or (350<=l or 360<=l):
        url += 'II/1.2_mosaics_v3.5/'
    if 11<=l and l<32:
        url += 'I/1.2_mosaics_v3.5/GLON_10-30/'
    if 32<=l and l<54:
        url += 'I/1.2_mosaics_v3.5/GLON_30-53/'
    if 54<=l and l<68:
        url += 'I/1.2_mosaics_v3.5/GLON_53-66/'
    if 284<=l and l<311:
        url += 'I/1.2_mosaics_v3.5/GLON_284_295-310/'
    if 311<=l and l<330:
        url += 'I/1.2_mosaics_v3.5/GLON_310-330/'
    if 330<=l and l<350:
        url += 'I/1.2_mosaics_v3.5/GLON_330-350/'
        
    func = lambda l: (l//3+1)*3 if l%3 > 1.5 else (l//3)*3 
    l = str(func(l))+'00'
    while len(l)!=5:
        l = '0'+l
        
    file_name = 'GLM_'+l+'+0000_mosaic_'+band_dict[band]+'.fits'
    with urllib.request.urlopen(url+file_name) as u:
        with open(save_path/file_name, 'bw') as o:
            o.write(u.read())
            
    return

def download_sage_data(band, save_path):
    '''
    band must be str, irac1, 2, 3, 4 or mips24.
    '''
    instrument_dict = {
        'irac': 'irac/v2.2_8x8_2.0_images/',
        'mips': 'mips_full_mos/',
    }
    
    band_dict = {
        'irac1': 'SAGE_LMC_IRAC3.6_2_mosaic.fits',
        'irac2': 'SAGE_LMC_IRAC4.5_2_mosaic.fits',
        'irac3': 'SAGE_LMC_IRAC5.8_2_mosaic.fits',
        'irac4': 'SAGE_LMC_IRAC8.0_2_mosaic.fits',
        'mips24': 'SAGE_LMC_MIPS24_E2.fits',
    }
    save_path = pathlib.Path(save_path).expanduser()
    url = 'https://irsa.ipac.caltech.edu//data/SPITZER/SAGE/images/'
    url += instrument_dict[band[:4]]
    
    file_name = band_dict[band]
    with urllib.request.urlopen(url+file_name) as u:
        with open(save_path/file_name, 'bw') as o:
            o.write(u.read())
            
    return

def get_fits_paths(l):
    '''
    l must be int (recommended multiple of 3) from 0 to 359.
    '''
    l = (lambda l: (l//3+1)*3 if l%3 > 1.5 else (l//3)*3)(l)
    path_mips = pathlib.Path('../data/raw/mipsgal/').expanduser().resolve()
    path_glim = pathlib.Path('../data/raw/glimpse/').expanduser().resolve() 
    
    paths_mips = [
        path_mips/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l-1)))+str(l-1)),
        path_mips/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l-1)))+str(l-1)),
        path_mips/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l)))+str(l)),
        path_mips/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l)))+str(l)),
        path_mips/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l+1)))+str(l+1)),
        path_mips/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l+1)))+str(l+1)),
    ]
    
    paths_glim = [
        path_glim/'irac1/GLM_{}00+0000_mosaic_I1.fits'.format('0'*(3-len(str(l)))+str(l)),
        path_glim/'irac4/GLM_{}00+0000_mosaic_I4.fits'.format('0'*(3-len(str(l)))+str(l)),
    ]
    
    if l == 0:
        paths_mips[0] = path_mips/'mips24/MG3590n005_024.fits'
        paths_mips[1] = path_mips/'mips24/MG3590p005_024.fits'
    else:
        pass
    
    return paths_mips, paths_glim

def _nparr_sum(nparr1, nparr2):
    r1 = numpy.nan_to_num(nparr1)
    r2 = numpy.nan_to_num(nparr2)
    r1_bool = numpy.where(r1==0, False, True)
    r2_bool = numpy.where(r2==0, False, True)
    r_bool = numpy.logical_and(r1_bool, r2_bool)
    r = numpy.where(r_bool==True, (r1+r2)/2, r1+r2)
    return r

def _new_header(header):
    header.pop('HISTORY*')
    header.pop('N2HASH')
    return header

def make_rgb_fits(paths_mips, paths_glim, save_path='../data/interim/gal'):
    save_path = pathlib.Path(save_path).expanduser().resolve()
    save_path = save_path/('spitzer_' + paths_glim[1].name[4:14] + '_rgb')
    save_path.mkdir(parents=True, exist_ok=True)
    
    rs_hdu_raw = [n2.open_fits(i) for i in paths_mips]
    g_hdu_raw = n2.open_fits(paths_glim[1])
    b_hdu_raw = n2.open_fits(paths_glim[0])
    
    header = g_hdu_raw.hdu.header.copy()
    rs_hdu = [i.regrid(header) for i in rs_hdu_raw]
    #g_hdu = g_hdu_raw.regrid(header)
    g_hdu = g_hdu_raw
    b_hdu = b_hdu_raw.regrid(header)
    
    r1 = _nparr_sum(rs_hdu[0].data, rs_hdu[1].data)
    r2 = _nparr_sum(rs_hdu[2].data, rs_hdu[3].data)
    r3 = _nparr_sum(rs_hdu[4].data, rs_hdu[5].data)
    r4 = _nparr_sum(r1, r2)
    r = _nparr_sum(r3, r4)    
    g = numpy.nan_to_num(g_hdu.data)
    b = numpy.nan_to_num(b_hdu.data)
    
    r_hdu = astropy.io.fits.PrimaryHDU(r, _new_header(rs_hdu[0].header))
    g_hdu = astropy.io.fits.PrimaryHDU(g, _new_header(g_hdu.header))
    b_hdu = astropy.io.fits.PrimaryHDU(b, _new_header(b_hdu.header))    
    r_hdu_list = astropy.io.fits.HDUList([r_hdu])
    g_hdu_list = astropy.io.fits.HDUList([g_hdu])
    b_hdu_list = astropy.io.fits.HDUList([b_hdu])
    r_hdu_list.writeto(save_path/'r.fits', overwrite=True)
    g_hdu_list.writeto(save_path/'g.fits', overwrite=True)
    b_hdu_list.writeto(save_path/'b.fits', overwrite=True)
    
    return

#######################
###=== tmp start ===###
#######################

def get_bubble_table():
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

def _get_dir(sample, path):
    files = list(pathlib.Path(path).expanduser().glob('*'))
    over_358p5 = sample.loc[:, 'l']>358.5
    for i in sample[over_358p5].loc[:, 'l'].index:
        sample.loc[i, 'l'] -= 360

    for file in files:
        file = str(file).split('/')[-1]
        l_center = float(file[8:11])
        l_min = l_center - 1.5
        l_max = l_center + 1.5
        _slice = (sample.loc[:, 'l']>=l_min)&(sample.loc[:, 'l']<l_max)
        for i in sample.loc[_slice].index:
            sample.loc[i, 'directory'] = file
            pass
        pass

    under_0p0 = sample.loc[:, 'l']<0
    for i in sample[under_0p0].loc[:, 'l'].index:
        sample.loc[i, 'l'] += 360

    # drop NaN file line
    sample = sample.dropna(subset=['directory'])
    return sample

def get_sample_table(bub=get_bubble_table(), fac=10, R_range = [0, 10]):
    l = [i for i in range(0, 66, 3)] + [i for i in range(297, 360, 3)]
    b = [-0.8, 0.8] # deg
    R = [0.1, 10] # arcmin
    l_bub = bub.loc[:, 'l'].tolist()
    b_bub = bub.loc[:, 'b'].tolist()
    R_bub = bub.loc[:, 'Rout'].tolist()
    # Generate coordinates and size randomly within specified range    
    name, glon_li, glat_li, size_li, i_n = [], [], [], [], 1
    while len(glon_li) < len(bub)*fac:
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
        else: pass
    nbub = pandas.DataFrame({'name': name, 'l': glon_li, 'b': glat_li, 'Rout': size_li})
    nbub = nbub.set_index('name')
    # add columns for label
    bub = bub.assign(label=1)
    nbub = nbub.assign(label=0)
    sample = bub.append(nbub)[bub.columns.tolist()]
    sample = sample.loc[(sample.loc[:, 'Rout']>R_range[0])&(sample.loc[:, 'Rout']<R_range[1])]    
    sample = _get_dir(sample, path='~/jupyter/spitzer_bubble/data/interim/gal')
    return sample

#####################
###=== tmp end ===###
#####################