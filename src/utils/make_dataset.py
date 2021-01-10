# import sys
import copy
import pathlib
import numpy
import pandas
import json
import pickle
from tensorflow.keras import utils as np_utils
from . import file_utils


###=== Dataset 作成の設定 ===
ring_file = '003-ring_50x50'
ring_aug_file = '004-ring_50x50_aug20_rot_ref'
nring_file = '005-nring_50x50'

test_ring_need = [
    'N2', 'N18', 'N64', 'N69', 'N76',
    'N81', 'S6', 'S13', 'S57', 'S82',
    'S113', 'S114', 'S186', 'CN52', 'CN99',
    'CN61',
]

bad_ring = [
    'CN143', 'N5', 'N19', 'N63', 'N109',
    'N128', 'N129', 'CS87', 'CS89', 'CS97',
    'S12', 'S17', 'S25', 'S44', 'S59',
    'S61', 'S73', 'S74', 'S77', 'S122',
    'S158', 'S167', 'S183', 'S184', 'S185',
    'S188', 'CN49', 'CN51', 'CS61', 'CS86',
    'S101', 'S153', 'S154', 'CN96', 'N120',
    'S93', 'S131', 'CS90', 'CN146',
]

ring_path = pathlib.Path(
    '~/jupyter/spitzer_bubble/data/interim/dataset/ring'
).expanduser()

nring_path = pathlib.Path(
    '~/jupyter/spitzer_bubble/data/interim/dataset/nring'
).expanduser()
###==========================



def test(validation_area):
    '''
    後に機能を分離するべきかもしれない
    '''
    ### numpy data を開く
    ring_cube = numpy.load(ring_path/ring_file/'ring.npy')
    ring_aug_cube = numpy.load(ring_path/ring_aug_file/'ring_aug.npy')
    nring_cube = numpy.load(nring_path/nring_file/'nring.npy')

    ### リングの情報を取得
    ring_infos = file_utils.open_json(ring_path/ring_file/'ring.json')
    ring_aug_infos = file_utils.open_json(ring_path/ring_aug_file/'ring_aug.json')
    nring_infos = file_utils.open_json(nring_path/nring_file/'nring.json')
    ring_df = pandas.DataFrame(ring_infos).set_index('name').drop(bad_ring)
    ring_aug_df = pandas.DataFrame(ring_aug_infos).set_index('name').drop(bad_ring)
    nring_df = pandas.DataFrame(nring_infos).set_index('name')

    ### ゴミリングを除く
    bad_nring = []
    datas = nring_cube
    for i, data_ in enumerate(datas):
        if data_.max()==data_.max():
            pass

        else: 
            bad_nring.append(nring_infos[i]['name'])
            pass
        pass

    nring_df = pandas.DataFrame(nring_infos).set_index('name').drop(bad_nring)

    ### test 領域を除く
    ring_df = ring_df.loc[ring_df.loc[:, 'directory']!=validation_area]
    ring_aug_df = ring_aug_df.loc[ring_aug_df.loc[:, 'directory']!=validation_area]
    nring_df = nring_df.loc[nring_df.loc[:, 'directory']!=validation_area]

    ### 数を出しておく
    ring_num = len(ring_df)
    nring_num = len(nring_infos)

    train_ring_num = int(ring_num*0.8)
    train_nring_num = int(nring_num*0.8)

    ### 上記の数を基にランダムにサンプリングする
    train_ring = ring_df.drop(test_ring_need, errors='ignore').sample(n=train_ring_num).index
    train_nring = nring_df.sample(n=train_nring_num).index
    test_ring = ring_df.drop(train_ring, errors='ignore').index
    test_nring = nring_df.drop(train_nring, errors='ignore').index

    # train_ring, test_ringのDataFrameを作る
    train_ring_df = ring_aug_df.loc[train_ring].sort_values('npy_index')
    test_ring_df = ring_df.loc[test_ring].sort_values('npy_index')

    # train_nring, test_nringのDataFrameを作る
    train_nring_df = nring_df.loc[train_nring].sort_values('npy_index')
    test_nring_df = nring_df.loc[test_nring].sort_values('npy_index')

    # train と比率を揃えるためにランダムに選んで数を減らす
    test_nring_df = test_nring_df.sample(n=int(len(test_ring_df)/(len(train_ring_df)/len(train_nring_df))))

    # train_ring, test_ringのindexを取得
    train_ring_indices = train_ring_df.loc[:, 'npy_index']
    test_ring_indices = test_ring_df.loc[:, 'npy_index']

    # train_nring, test_nringのindexを取得
    train_nring_indices = train_nring_df.loc[:, 'npy_index']
    test_nring_indices = test_nring_df.loc[:, 'npy_index']

    # train_ring, test_ringのdataを切り出す
    train_ring_cube = ring_aug_cube[train_ring_indices]
    test_ring_cube = ring_cube[test_ring_indices]

    # train_nring, test_nringのdataを切り出す
    train_nring_cube = nring_cube[train_nring_indices]
    test_nring_cube = nring_cube[test_nring_indices]

    x_train = numpy.concatenate([train_ring_cube, train_nring_cube], axis=0)
    x_test = numpy.concatenate([test_ring_cube, test_nring_cube], axis=0)
    y_train = numpy.array(list(train_ring_df.loc[:, 'label']) + list(train_nring_df.loc[:, 'label']))
    y_test = numpy.array(list(test_ring_df.loc[:, 'label']) + list(test_nring_df.loc[:, 'label']))

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    train_df = pandas.concat([train_ring_df, train_nring_df]).reset_index()
    train_df = train_df.where(train_df==train_df, None)
    test_df = pandas.concat([test_ring_df, test_nring_df]).reset_index()
    test_df = test_df.where(test_df==test_df, None)

    train_infos = train_df.reset_index().to_dict('records')
    test_infos = test_df.reset_index().to_dict('records')

    save_path = pathlib.Path(
        '~/jupyter/spitzer_bubble/data/processed/dataset/all/'
    ).expanduser()/validation_area
    save_path.mkdir(parents=True)
    numpy.save(save_path/'x_train.npy', x_train)
    numpy.save(save_path/'y_train.npy', y_train)
    numpy.save(save_path/'x_test.npy', x_test)
    numpy.save(save_path/'y_test.npy', y_test)
    
    train_df = pandas.concat([train_ring_df, train_nring_df]).reset_index()
    train_df = train_df.where(train_df==train_df, None)
    test_df = pandas.concat([test_ring_df, test_nring_df]).reset_index()
    test_df = test_df.where(test_df==test_df, None)
    train_infos = train_df.reset_index().to_dict('records')
    test_infos = test_df.reset_index().to_dict('records')
    
    file_utils.save_json(save_path/'train.json', train_infos)
    file_utils.save_json(save_path/'test.json', test_infos)
    
    pass