import pathlib
import urllib.request

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
