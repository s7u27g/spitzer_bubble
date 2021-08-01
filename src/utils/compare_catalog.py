import numpy as np
import pandas as pd

def find_pos_candidate(se, df, pos_th):
    x_df, x_se = df.loc[:,'x'], se.loc['x']
    y_df, y_se = df.loc[:,'y'], se.loc['y']
    r_df, r_se = df.loc[:,'r'], se.loc['r']
    distance = ((x_df - x_se)**2 + (y_df - y_se)**2)**(1/2)*60
    mask = distance<r_df*(pos_th)
    if np.nansum(mask)==0: return []
    else: return list(df[mask].index)

def find_size_candidate(se, df, size_th):
    r_df, r_se = df.loc[:,'r'], se.loc['r']
    r_fac = r_df.copy()
    r_fac[r_df>r_se] = r_se/r_df
    r_fac[r_df<r_se] = r_df/r_se
    r_fac[r_df==r_se] = 1
    mask = r_fac>size_th
    if np.nansum(mask)==0: return []
    else: return list(df[mask].index)

def select_match_obj(se, df):
    r_df, r_se = df.loc[:,'r'], se.loc['r']
    r_fac = r_df.copy()
    r_fac[r_df>r_se] = r_se/r_df
    r_fac[r_df<r_se] = r_df/r_se
    r_fac[r_df==r_se] = 1
    return r_fac.idxmax()


def find_overlap_obj(df, pos_th, size_th):
    '''
    return: double list (match pair)
    '''
    ### 重複ありの集合リスト
    overlap_tmp1 = []
    for i, se in df.iterrows():
        obj = find_pos_candidate(se, df, pos_th)
        obj = find_size_candidate(se, df.loc[obj], size_th)
        if len(obj)==1: pass # 一つしかなければ自分自身のみ
        else: overlap_tmp1.append(set(obj)) # 自分自身も含んで
        pass

    ### 重複を除いた集合リスト
    ### ある集合の一部要素のみをもつ集合は含まれる可能性あり
    overlap_tmp2 = []
    for _ in overlap_tmp1:
        if not _ in overlap_tmp2:
            overlap_tmp2.append(_)
            pass
        pass

    ### ある集合の一部要素のみを持つ集合を除いた集合を list に変換
    overlap = []
    for over3 in [_ for _ in overlap_tmp2 if len(_)>2]:
        for _ in overlap_tmp2:
            if over3>_:
                pass
            else:
                overlap.append(list(_))
                pass
            pass
        pass

    return overlap


def find_match_obj(df1, df2, pos_th, size_th):
    '''
    Check which celestial body in df2 matches df1.
    return: new df1
    '''
    match_objs = {}
    for i, se in df1.iterrows():
        obj = find_pos_candidate(se, df2, pos_th)
        obj = find_size_candidate(se, df2.loc[obj], size_th)
        if len(obj) == 0:
            pass
        else:
            ### 最もサイズが近いものに絞る
            obj = select_match_obj(se, df2.loc[obj])
            match_objs[se.name] = obj # df1 と df2 のペア
            pass
        pass

    df2_objs = set(match_objs.values()) # df1 にある df2 の円

    ### df2 の円一つに対して一致候補 df1 のリストを作る
    overlap = {
        df2_obj: [match_obj[0] for match_obj in match_objs.items() if df2_obj==match_obj[1]]
        for df2_obj in df2_objs
    }

    ### 候補を一つに絞る
    match_objs = {}
    for df2_obj, df1_obj in overlap.items():
        match_objs[df2_obj] = select_match_obj(df2.loc[df2_obj], df1.loc[df1_obj])
        pass

    ### DataFrame を作る
    match_objs = [{'name': v, 'match_obj': k} for k, v in match_objs.items()]
    df_ = pd.DataFrame(match_objs).set_index('name')
    df1_new = pd.concat([df1, df_], axis=1, sort=True)

    return df1_new


def split_tp_fp_fn(df1, df2, pos_th, size_th, coord=None, verbose=False):
    '''
    Check which celestial body in df2 matches df1.
    return: dict of df
    '''
    if coord:
        df1 = df1.rename(columns={coord[0]: 'x', coord[1]: 'y', coord[2]: 'Rout'})
        df2 = df2.rename(columns={coord[0]: 'x', coord[1]: 'y', coord[2]: 'Rout'})
        pass

    df1 = df1.reset_index().set_index('name').rename(columns={'index': 'org_idx'})
    df2 = df2.reset_index().set_index('name').rename(columns={'index': 'org_idx'})
    df1 = find_match_obj(df1, df2, pos_th, size_th)
    tp_df1 = df1.loc[df1.loc[:,'match_obj']==df1.loc[:,'match_obj']]
    fp_df1 = df1.loc[df1.loc[:,'match_obj']!=df1.loc[:,'match_obj']]
    tp_df2 = df2.loc[df2.index.isin(df1.loc[:,'match_obj'].unique())]
    fn_df2 = df2.loc[~df2.index.isin(df1.loc[:,'match_obj'].unique())]

    if coord:
        tp_df1 = tp_df1.rename(columns={'x': coord[0], 'y': coord[1], 'r': coord[2]})
        fp_df1 = fp_df1.rename(columns={'x': coord[0], 'y': coord[1], 'r': coord[2]})
        tp_df2 = tp_df2.rename(columns={'x': coord[0], 'y': coord[1], 'r': coord[2]})
        fn_df2 = fn_df2.rename(columns={'x': coord[0], 'y': coord[1], 'r': coord[2]})
        pass

    ret = {
        'tp_df1': tp_df1.reset_index().rename(columns={'index': 'name'}),
        'fp_df1': fp_df1.reset_index().rename(columns={'index': 'name'}),
        'tp_df2': tp_df2.reset_index().rename(columns={'index': 'name'}),
        'fn_df2': fn_df2.reset_index().rename(columns={'index': 'name'}),
    }

    if verbose:
        tp, fp, fn = len(tp_df1), len(fp_df1), len(fn_df2)
        print(f'  網羅率: {round(tp/(tp+fn)*100, 2)} %')
        print(f'誤検知率: {round(fp/(tp+fp)*100, 2)} %')
        pass

    return ret
