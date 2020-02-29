import urllib.request

def download_mipsgal_data(l, b, save_path='/root/data/science/spitzer/mipsgal/'):
    '''
    l must be int.
    b must be str, 'p' or 'n'.
    '''
    url = 'https://irsa.ipac.caltech.edu/data/SPITZER/MIPSGAL/images/mosaics24/'
    l = str(l)+'0'
    while len(l)!=4:
        l = '0'+l
        
    file_name = 'MG'+l+b+'005_024.fits'
    with urllib.request.urlopen(url+file_name) as u:
        with open(save_path+file_name, 'bw') as o:
            o.write(u.read())
            
    return

