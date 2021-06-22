import os
import numpy as np
import pandas as pd


roi_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/RoiSlice'
label_df = pd.DataFrame(columns=['PZ', 'CG', 'U', 'AMSF'])

for roi in os.listdir(roi_folder):
    roi_path = os.path.join(roi_folder, roi)
    case_name = roi[: roi.index('.npy')]
    roi_arr = np.load(roi_path)

    roi_label = np.array([np.sum(roi_arr[1, ...]), np.sum(roi_arr[2, ...]), np.sum(roi_arr[3, ...]), np.sum(roi_arr[4, ...])], dtype=np.int)
    roi_label[roi_label > 0] = 1
    case_df = pd.DataFrame(data=[roi_label], index=[case_name], columns=['PZ', 'CG', 'U', 'AMSF'])

    label_df = label_df.append(case_df)

label_df.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/class_label.csv')




