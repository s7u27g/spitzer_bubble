import pathlib
import urllib.request
import numpy
import astropy.io.fits
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
