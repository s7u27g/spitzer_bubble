import astroquery.vizier
import pandas as pd

def churchwell_bubble():
    # make instance
    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    # load bub_2006
    bub_2006 = viz.query_constraints(catalog='J/ApJ/649/759/bubbles')[0].to_pandas()
#     bub_2006.loc[:, '__CPA2006_'] = bub_2006.loc[:, '__CPA2006_'].str.decode('utf-8')
#     bub_2006.loc[:, 'MFlags'] = bub_2006.loc[:, 'MFlags'].str.decode('utf-8')
    # load bub_2007
    bub_2007 = viz.query_constraints(catalog='J/ApJ/670/428/bubble')[0].to_pandas()
#     bub_2007.loc[:, '__CWP2007_'] = bub_2007.loc[:, '__CWP2007_'].str.decode('utf-8')
#     bub_2007.loc[:, 'MFlags'] = bub_2007.loc[:, 'MFlags'].str.decode('utf-8')
    # convert to pandas for 2006
    bub_2006.rename(columns={'__CPA2006_': 'name'}, inplace=True)
    bub_2006.rename(columns={'GLON': 'l'}, inplace=True)
    bub_2006.rename(columns={'GLAT': 'b'}, inplace=True)
    bub_2006.rename(columns={'__R_': 'Reff'}, inplace=True)
    bub_2006.rename(columns={'__T_': 'Teff'}, inplace=True)
    bub_2006.rename(columns={'MFlags': 'Flags'}, inplace=True)
    bub_2006.rename(columns={'_RA.icrs': 'RA.icrs'}, inplace=True)
    bub_2006.rename(columns={'_DE.icrs': 'DE.icrs'}, inplace=True)
    bub_2006 = bub_2006.set_index('name')
    # convert to pandas for 2007
    bub_2007.rename(columns={'__CWP2007_': 'name'}, inplace=True)
    bub_2007.rename(columns={'GLON': 'l'}, inplace=True)
    bub_2007.rename(columns={'GLAT': 'b'}, inplace=True)
    bub_2007.rename(columns={'__R_': 'Reff'}, inplace=True)
    bub_2007.rename(columns={'__T_': 'Teff'}, inplace=True)
    bub_2007.rename(columns={'MFlags': 'Flags'}, inplace=True)
    bub_2007.rename(columns={'_RA.icrs': 'RA.icrs'}, inplace=True)
    bub_2007.rename(columns={'_DE.icrs': 'DE.icrs'}, inplace=True)
    for i in bub_2007.index:
        bub_2007.loc[i, 'name'] = bub_2007.loc[i, 'name'].replace(' ', '')
        pass
    bub_2007 = bub_2007.set_index('name')
    # concat 2006 and 2007
    bub = pd.concat([bub_2006, bub_2007])
    return bub.reset_index()

def mwp1st_bubble():
    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    mwp_small = viz.query_constraints(catalog='J/MNRAS/424/2442/mwpsmall')[0].to_pandas()
#     mwp_small.loc[:, 'MWP'] = mwp_small.loc[:, 'MWP'].str.decode('utf-8')
#     mwp_small.loc[:, 'ONames'] = mwp_small.loc[:, 'ONames'].str.decode('utf-8')
#     mwp_small.loc[:, 'Simbad'] = mwp_small.loc[:, 'Simbad'].str.decode('utf-8')
    mwp_small = mwp_small.drop(columns='Flag')
    #mwp_small.loc[:, 'Flag'] = mwp_small.loc[:, 'Flag'].str.decode('utf-8')
    mwp_large = viz.query_constraints(catalog='J/MNRAS/424/2442/mwplarge')[0].to_pandas()
#     mwp_large.loc[:, 'MWP'] = mwp_large.loc[:, 'MWP'].str.decode('utf-8')
#     mwp_large.loc[:, 'ONames'] = mwp_large.loc[:, 'ONames'].str.decode('utf-8')
#     mwp_large.loc[:, 'Simbad'] = mwp_large.loc[:, 'Simbad'].str.decode('utf-8')
    mwp_large = mwp_large.drop(columns='Flag')
    #mwp_large.loc[:, 'Flag'] = mwp_large.loc[:, 'Flag'].str.decode('utf-8')
    mwp_small.rename(columns={'MWP': 'name'}, inplace=True)
    mwp_small.rename(columns={'GLON': 'l'}, inplace=True)
    mwp_small.rename(columns={'GLAT': 'b'}, inplace=True)
    # mwp_small.rename(columns={'Reff': 'Rout'}, inplace=True)
    mwp_large.rename(columns={'MWP': 'name'}, inplace=True)
    mwp_large.rename(columns={'GLON': 'l'}, inplace=True)
    mwp_large.rename(columns={'GLAT': 'b'}, inplace=True)
    # mwp_large.rename(columns={'Reff': 'Rout'}, inplace=True)

    mwp_small = mwp_small.set_index('name')
    mwp_large = mwp_large.set_index('name')
    bub = pd.concat([mwp_small, mwp_large], axis=0, sort=False)
    return bub.reset_index()

def mwp2nd_bubble():
    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    mwp = viz.query_constraints(catalog='J/MNRAS/488/1141/table3')[0].to_pandas()
    mwp.rename(columns={'MWP': 'name'}, inplace=True)
    mwp.rename(columns={'GLON': 'l'}, inplace=True)
    mwp.rename(columns={'GLAT': 'b'}, inplace=True)
    # mwp.rename(columns={'Reff': 'Rout'}, inplace=True)
    mwp = mwp.set_index('name')
    return mwp.reset_index()

def wise_hii():
    viz = astroquery.vizier.Vizier(columns=['*'])
    viz.ROW_LIMIT = -1
    df = viz.query_constraints(catalog='J/ApJS/212/1/wisecat')[0].to_pandas()
#     df.loc[:, 'WISE'] = df.loc[:, 'WISE'].str.decode('utf-8')
#     df.loc[:, 'Cl'] = df.loc[:, 'Cl'].str.decode('utf-8')
#     df.loc[:, 'Ref'] = df.loc[:, 'Ref'].str.decode('utf-8')
#     df.loc[:, 'Name'] = df.loc[:, 'Name'].str.decode('utf-8')
#     df.loc[:, 'Mol'] = df.loc[:, 'Mol'].str.decode('utf-8')
    df.rename(columns={'WISE': 'name'}, inplace=True)
    df.rename(columns={'GLON': 'l'}, inplace=True)
    df.rename(columns={'GLAT': 'b'}, inplace=True)
    # df.rename(columns={'Rad': 'Rout'}, inplace=True)
    df.rename(columns={'Name': 'Hii_name'}, inplace=True)
    # df.loc[:, 'Rout'] = df.loc[:, 'Rout']/60
    df.loc[:, 'Rad'] = df.loc[:, 'Rad']/60
    df = df.set_index('name')
    return df.reset_index()

def hou_hii():
    viz = astroquery.vizier.Vizier(
        columns=['Seq', 'N', 'GLON', 'GLAT', 'VLSR', 'Dist', 'u_Dist', 'n_Dist', 'R8.5', 'D8.5', 'U8.5']
    )
    viz.ROW_LIMIT = -1
    df = viz.query_constraints(catalog='J/A+A/569/A125/tablea1')[0].to_pandas()
    df.rename(columns={'GLON': 'l'}, inplace=True)
    df.rename(columns={'GLAT': 'b'}, inplace=True)

    return df
