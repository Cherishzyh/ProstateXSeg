import os
import numpy as np
import pandas as pd
from random import shuffle


def DivideDataBySlice():
    '''
    train : val : test = 0.64 : 0.16 : 0.2 = 1175 : 294: 367
    # num should divided by 24
    # train : val : test = 1200 : 312 : 324 ≈ 0.65 : 0.17 : 0.18
    :return:
    '''
    data_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_list = os.listdir(data_folder)
    shuffle(data_list)
    test_num = int(len(data_list) * 0.2)
    train_num = int((len(data_list) - test_num) * 0.8)
    val_num = len(data_list) - test_num - train_num
    train_name, val_name, test_name = [], [], []
    for case in data_list:
        case = case[: case.index('.npy')]
        if len(train_name) < train_num:
            train_name.append(case)
        elif len(val_name) < val_num:
            val_name.append(case)
        else:
            test_name.append(case)
    # print(train_num, val_num, test_num)
    print(len(train_name), len(val_name), len(test_name))

    train_df = pd.DataFrame(train_name)
    val_df = pd.DataFrame(val_name)
    test_df = pd.DataFrame(test_name)
    train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/train_name.csv', index=False)
    val_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/val_name.csv', index=False)
    test_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/test_name.csv', index=False)
# DivideDataBySlice()


def DivideDataByCase():
    '''
    train : val : test = 0.64 : 0.16 : 0.2 = 62: 16: 20
    '''
    data_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_list = sorted(os.listdir(data_folder))
    case_list = [case[: case.index('_-_')] for case in data_list]
    case_list = list(set(case_list))
    # shuffle(case_list)
    # test_num = int(len(case_list) * 0.2)
    # train_num = int((len(case_list) - test_num) * 0.8)
    # val_num = len(case_list) - test_num - train_num
    # train_case, val_case, test_case = [], [], []
    # for case in case_list:
    #     if len(train_case) < train_num:
    #         train_case.append(case)
    #     elif len(val_case) < val_num:
    #         val_case.append(case)
    #     else:
    #         test_case.append(case)
    # # print(train_num, val_num, test_num)
    # print(len(train_case), len(val_case), len(test_case))
    # train_df = pd.DataFrame(train_case)
    # val_df = pd.DataFrame(val_case)
    # test_df = pd.DataFrame(test_case)
    # train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/train_case_name.csv', index=False)
    # val_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/val_case_name.csv', index=False)
    # test_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/test_case_name.csv', index=False)

    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    train_df = pd.read_csv(os.path.join(data_root, 'train_case_name.csv'))
    train_case = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_case_name.csv'))
    val_case = val_df.values.tolist()[0]
    test_df = pd.read_csv(os.path.join(data_root, 'test_case_name.csv'))
    test_case = test_df.values.tolist()[0]

    train_name, val_name, test_name = [], [], []
    for case in case_list:
        if case in train_case:
            train_name.extend([slice[: slice.index('.npy')] for slice in data_list if case in slice])
        elif case in val_case:
            val_name.extend([slice[: slice.index('.npy')] for slice in data_list if case in slice])
        elif case in test_case:
            test_name.extend([slice[: slice.index('.npy')] for slice in data_list if case in slice])

    print(len(train_name), len(val_name), len(test_name))
    train_df = pd.DataFrame(train_name)
    val_df = pd.DataFrame(val_name)
    test_df = pd.DataFrame(test_name)
    train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/train_name.csv', index=False)
    val_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/val_name.csv', index=False)
    test_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/test_name.csv', index=False)
# DivideDataByCase()


def RoiSta():
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]
    test_df = pd.read_csv(os.path.join(data_root, 'test_name.csv'))
    test_list = test_df.values.tolist()[0]

    case_path = os.path.join(data_root, 'RoiSlice')
    pz_train, cg_train, u_train, af_train = 0, 0, 0, 0
    pz_val, cg_val, u_val, af_val = 0, 0, 0, 0
    pz_test, cg_test, u_test, af_test = 0, 0, 0, 0
    for case in os.listdir(case_path):
        case_name = case[:case.index('.npy')]
        roi = np.squeeze(np.load(os.path.join(case_path, case)))
        if case_name in train_list:
            if np.sum(roi[1, ...]) != 0:
                pz_train += 1
            if np.sum(roi[2, ...]) != 0:
                cg_train += 1
            if np.sum(roi[3, ...]) != 0:
                u_train += 1
            if np.sum(roi[4, ...]) != 0:
                af_train += 1
        elif case_name in val_list:
            if np.sum(roi[1, ...]) != 0:
                pz_val += 1
            if np.sum(roi[2, ...]) != 0:
                cg_val += 1
            if np.sum(roi[3, ...]) != 0:
                u_val += 1
            if np.sum(roi[4, ...]) != 0:
                af_val += 1
        elif case_name in test_list:
            if np.sum(roi[1, ...]) != 0:
                pz_test += 1
            if np.sum(roi[2, ...]) != 0:
                cg_test += 1
            if np.sum(roi[3, ...]) != 0:
                u_test += 1
            if np.sum(roi[4, ...]) != 0:
                af_test += 1

    print(pz_train, cg_train, u_train, af_train)
    print(pz_val, cg_val, u_val, af_val)
    print(pz_test, cg_test, u_test, af_test)
# RoiSta()


def DivideDataByROI34():
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]
    test_df = pd.read_csv(os.path.join(data_root, 'test_name.csv'))
    test_list = test_df.values.tolist()[0]
    shuffle(train_list), shuffle(val_list), shuffle(test_df)
    case_path = os.path.join(data_root, 'RoiSlice')
    train_list_left = []
    val_list_left = []
    test_list_left = []
    for case in train_list:
        roi = np.squeeze(np.load(os.path.join(case_path, '{}.npy'.format(case))))
        if np.sum(roi[3, ...]) != 0 or np.sum(roi[4, ...]) != 0:
            train_list_left.append(case)
    for case in train_list:
        if len(train_list_left) < 900:
            if case not in train_list_left:
                train_list_left.append(case)
            else:
                continue
        else:
            break

    for case in val_list:
        roi = np.squeeze(np.load(os.path.join(case_path, '{}.npy'.format(case))))
        if np.sum(roi[3, ...]) != 0 or np.sum(roi[4, ...]) != 0:
            val_list_left.append(case)
    for case in val_list:
        if len(val_list_left) < 225:
            if case not in val_list_left:
                val_list_left.append(case)
            else:
                continue
        else:
            break

    for case in test_list:
        roi = np.squeeze(np.load(os.path.join(case_path, '{}.npy'.format(case))))
        if np.sum(roi[3, ...]) != 0 or np.sum(roi[4, ...]) != 0:
            test_list_left.append(case)
    for case in test_list:
        if len(test_list_left) < 281:
            if case not in test_list_left:
                test_list_left.append(case)
            else:
                continue
        else:
            break

    print(len(train_list_left), len(val_list_left), len(test_list_left))

    train_df = pd.DataFrame(train_list_left)
    val_df = pd.DataFrame(val_list_left)
    test_df = pd.DataFrame(test_list_left)
    train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/train_name_left.csv', index=False)
    val_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/val_name_left.csv', index=False)
    test_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/test_name_left.csv', index=False)

# DivideDataByROI34()


def RoiStaV2():
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]
    test_df = pd.read_csv(os.path.join(data_root, 'test_name.csv'))
    test_list = test_df.values.tolist()[0]

    case_path = os.path.join(data_root, 'RoiSlice')
    # pz_train, cg_train, u_train, af_train = 0, 0, 0, 0
    # pz_val, cg_val, u_val, af_val = 0, 0, 0, 0
    # pz_test, cg_test, u_test, af_test = 0, 0, 0, 0
    train_total, val_total, test_total = 0, 0, 0
    for case in os.listdir(case_path):
        case_name = case[:case.index('.npy')]
        roi = np.squeeze(np.load(os.path.join(case_path, case)))
        if case_name in train_list:
            if np.sum(roi[3, ...]) != 0 and np.sum(roi[4, ...]) != 0:
                train_total += 1
        elif case_name in val_list:
            if np.sum(roi[3, ...]) != 0 and np.sum(roi[4, ...]) != 0:
                val_total += 1
        elif case_name in test_list:
            if np.sum(roi[3, ...]) != 0 and np.sum(roi[4, ...]) != 0:
                test_total += 1

    print(train_total, val_total, test_total)
    # print(pz_val, cg_val, u_val, af_val)
    # print(pz_test, cg_test, u_test, af_test)
# RoiStaV2()

