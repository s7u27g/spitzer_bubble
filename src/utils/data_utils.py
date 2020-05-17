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

def make_mipsgal_293data(b, save_path):
    '''
    b must be str, 'p' or 'n'.
    '''
    path = pathlib.Path(save_path)    
    file_dict = {
        0: (path/'MG2930{}005_024.fits'.format(b)).expanduser(),
        1: (path/'MG2940{}005_024.fits'.format(b)).expanduser(),
        2: (path/'MG2950{}005_024.fits'.format(b)).expanduser(),
    }
    files = file_dict[1], file_dict[2]
    hdus = astropy.io.fits.open(files[0])[0], astropy.io.fits.open(files[1])[0]
    
    header, data = hdus[0].header, hdus[0].data
    d_l = hdus[1].header['CRVAL1'] - hdus[0].header['CRVAL1']
    
    header['CRVAL1'] = header['CRVAL1'] - d_l
    header['FILENAME'] = file_dict[0].name
    data = numpy.nan_to_num(data)
    data = numpy.where(data>0.0, 0.0, 0.0)
    data = data.astype(numpy.float32)
    new_hdu = astropy.io.fits.PrimaryHDU(data, header)
    new_hdul = astropy.io.fits.HDUList([new_hdu])
    new_hdul.writeto(file_dict[0], overwrite=True)
    
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

def get_mipsfits_paths(l):
    l = (lambda l: (l//3+1)*3 if l%3 > 1.5 else (l//3)*3)(l)
    path = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/raw/download/mipsgal/'
    ).expanduser().resolve()
    paths_mips = [
        path/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l-1)))+str(l-1)),
        path/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l-1)))+str(l-1)),
        path/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l)))+str(l)),
        path/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l)))+str(l)),
        path/'mips24/MG{}0n005_024.fits'.format('0'*(3-len(str(l+1)))+str(l+1)),
        path/'mips24/MG{}0p005_024.fits'.format('0'*(3-len(str(l+1)))+str(l+1)),
    ]
    
    if l == 0:
        paths_mips[0] = path/'mips24/MG3590n005_024.fits'
        paths_mips[1] = path/'mips24/MG3590p005_024.fits'
    else:
        pass
    
    return paths_mips

def get_fits_paths(l):
    l = (lambda l: (l//3+1)*3 if l%3 > 1.5 else (l//3)*3)(l)
    path_mips24 = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/raw/download/mipsgal/mips24_concat'
    ).expanduser().resolve()
    path_irac4 = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/raw/download/glimpse/irac4'
    ).expanduser().resolve()
    path_irac1 = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/raw/download/glimpse/irac1'
    ).expanduser().resolve()
    
    paths = [
        path_mips24/'MPG_{}00+0000_mosaic_M1.fits'.format('0'*(3-len(str(l)))+str(l)),
        path_irac4/'GLM_{}00+0000_mosaic_I4.fits'.format('0'*(3-len(str(l)))+str(l)),
        path_irac1/'GLM_{}00+0000_mosaic_I1.fits'.format('0'*(3-len(str(l)))+str(l)),
    ]
    
    return paths

def montage(rf_paths, save_path, interim_path='.'):
    '''
    rf_paths: raw fits path list
    save_path:
    interim_path:
    '''
    _tmp = str(save_path).replace('/', '-').split('.')[0][1:]
    tmp_path = pathlib.Path(interim_path).expanduser().resolve()/('.montage/'+_tmp)
    (tmp_path/'raw').mkdir(exist_ok=True, parents=True)
    (tmp_path/'raw_proj').mkdir(exist_ok=True, parents=True)
    
    for i, file in enumerate(rf_paths):
        cp_from = file
        cp_to = tmp_path/'raw/{}.fits'.format(i)
        subprocess.run(['cp', cp_from, cp_to])
        pass
    
    subprocess.run(['mImgtbl', tmp_path/'raw', tmp_path/'images.tbl'])
    subprocess.run(['mMakeHdr', tmp_path/'images.tbl', tmp_path/'template.hdr', 'GAL'])
    for file in sorted((tmp_path/'raw').glob('*')):
        subprocess.run(['mProjectCube', file, tmp_path/'raw_proj'/(file.stem+'_proj.fits'), tmp_path/'template.hdr'])
    subprocess.run(['mImgtbl', tmp_path/'raw_proj', tmp_path/'resultimages.tbl'])
    subprocess.run(['mAdd', '-p', tmp_path/'raw_proj', tmp_path/'resultimages.tbl', tmp_path/'template.hdr', tmp_path/'result.fits'])
    
    subprocess.run(['mv', tmp_path/'result.fits', save_path])
    subprocess.run(['rm', '-r', tmp_path])
    return

def _concat_mipsfits(l):
    '''
    l: l must be int (recommended multiple of 3) from 0 to 359.
    '''
    path = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/raw/download/mipsgal/mips24_concat'
    ).expanduser().resolve()
    path.mkdir(exist_ok=True)    
    func = lambda l: (l//3+1)*3 if l%3 > 1.5 else (l//3)*3
    _l = str(func(l))+'00'
    while len(_l)!=5:
        _l = '0'+_l
    file_name = 'MPG_'+_l+'+0000_mosaic_M1'+'.fits'
    montage(get_mipsfits_paths(l), path/file_name, path)
    return

def concat_mipsfits(l_list):
    '''
    l_list: list of l (must be int (recommended multiple of 3) from 0 to 359).
    '''
    for l in l_list:
        _concat_mipsfits(l)
        pass
    return

def _new_header(header):
    header.pop('HISTORY*')
    header.pop('N2HASH')
    return header

def make_rgb_fits(paths, save_path='~/jupyter/spitzer_bubble/data/raw/regrid/gal'):
    save_path = pathlib.Path(save_path).expanduser().resolve()
    save_path = save_path/('spitzer_' + paths[1].name[4:14] + '_rgb')
    save_path.mkdir(parents=True, exist_ok=True)
    
    r_hdu_raw = n2.open_fits(paths[0])
    g_hdu_raw = n2.open_fits(paths[1])
    b_hdu_raw = n2.open_fits(paths[2])
    
    header = g_hdu_raw.hdu.header.copy()
    r_hdu = r_hdu_raw.regrid(header)
    g_hdu = g_hdu_raw
    b_hdu = b_hdu_raw.regrid(header)
    
    r_hdu = astropy.io.fits.PrimaryHDU(r_hdu.data, _new_header(r_hdu.header))
    g_hdu = astropy.io.fits.PrimaryHDU(g_hdu.data, _new_header(g_hdu.header))
    b_hdu = astropy.io.fits.PrimaryHDU(b_hdu.data, _new_header(b_hdu.header))    
    r_hdu_list = astropy.io.fits.HDUList([r_hdu])
    g_hdu_list = astropy.io.fits.HDUList([g_hdu])
    b_hdu_list = astropy.io.fits.HDUList([b_hdu])
    r_hdu_list.writeto(save_path/'r.fits', overwrite=True)
    g_hdu_list.writeto(save_path/'g.fits', overwrite=True)
    b_hdu_list.writeto(save_path/'b.fits', overwrite=True)
    
    return