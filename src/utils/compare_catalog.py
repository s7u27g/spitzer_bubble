import numpy as np
import pandas as pd

def find_pos_candidate(se, df, pos_th):
    l_df, l_se = df.loc[:,'l'], se.loc['l']
    b_df, b_se = df.loc[:,'b'], se.loc['b']
    R_df, R_se = df.loc[:,'Rout'], se.loc['Rout']
    distance = ((l_df - l_se)**2 + (b_df - b_se)**2)**(1/2)*60
    mask = distance<R_df*(pos_th)
    if np.nansum(mask)==0: return []
    else: return list(df[mask].index)
    
def find_size_candidate(se, df, size_th):
    R_df, R_se = df.loc[:,'Rout'], se.loc['Rout']
    R_fac = R_df.copy()
    R_fac[R_df>R_se] = R_se/R_df
    R_fac[R_df<R_se] = R_df/R_se
    R_fac[R_df==R_se] = 1
    mask = R_fac>size_th
    if np.nansum(mask)==0: return []
    else: return list(df[mask].index)
    
def select_match_obj(se, df):
    R_df, R_se = df.loc[:,'Rout'], se.loc['Rout']
    R_fac = R_df.copy()
    R_fac[R_df>R_se] = R_se/R_df
    R_fac[R_df<R_se] = R_df/R_se
    R_fac[R_df==R_se] = 1
    return R_fac.idxmax()


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
            match_objs[se.name] = obj # ML天体とMWP天体のペア
            pass
        pass

    mwp_objs = set(match_objs.values()) # ML で検出した MWP の天体

    ### MWP天体一つに対して一致候補ML天体のリストを作る
    overlap = {
        mwp_obj: [match_obj[0] for match_obj in match_objs.items() if mwp_obj==match_obj[1]]
        for mwp_obj in mwp_objs
    }

    ### 候補を一つに絞る
    match_objs = {}
    for mwp, mls in overlap.items():
        match_objs[mwp] = select_match_obj(df2.loc[mwp], df1.loc[mls])
        pass

    ### DataFrame を作る
    match_objs = [{'name': v, 'match_obj': k} for k, v in match_objs.items()]
    df_ = pd.DataFrame(match_objs).set_index('name')
    df1_new = pd.concat([df1, df_], axis=1, sort=True)
    
    return df1_new