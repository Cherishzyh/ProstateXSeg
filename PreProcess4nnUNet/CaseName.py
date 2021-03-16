import os
import shutil
import pandas as pd

OriginalDataPath = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
DataDivideCSV = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/test_case_name.csv'

DesDataPath = r'/home/zhangyihong/Documents/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_ProstateX'

train_path = os.path.join(DesDataPath, 'imagesTr')
test_path = os.path.join(DesDataPath, 'imagesTs')
label_path = os.path.join(DesDataPath, 'labelsTr')


def GetCaseID(case):
    id = case[-3:]
    new_name = 'ProstateX_{}'.format(id)
    return new_name

def StandardCaseID():
    test_df = pd.read_csv(DataDivideCSV)
    test_list = test_df.values.tolist()[0]
    case_list = os.listdir(OriginalDataPath)
    for case in case_list:
        case_folder = os.path.join(OriginalDataPath, case)
        t2_path = os.path.join(case_folder, 't2.nii')
        roi_path = os.path.join(case_folder, 'roi.nii.gz')
        # ProstateX_xxx(num)
        new_name = GetCaseID(case)

        #     t2: change ProstateX-0004 to ProstateX_xxx(num)_xxxx(mod.).nii.gz and copy to imagesTs
        #    roi: change ProstateX-0004 to ProstateX_xxx(num).nii.gz and copy to imagesTs
        # have no idea that whether the label of test should be save

        if case in test_list:
            des_t2_path = os.path.join(test_path, new_name+'_0000.nii.gz')
        else:
            des_t2_path = os.path.join(train_path, new_name+'_0000.nii.gz')

        des_label_path = os.path.join(label_path, new_name+'.nii.gz')
        print('###################copying {}###################'.format(case))
        shutil.copy(t2_path, des_t2_path)
        shutil.copy(roi_path, des_label_path)



if __name__ == '__main__':
    StandardCaseID()