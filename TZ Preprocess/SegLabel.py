import os
import numpy as np
import SimpleITK as sitk
from MeDIT.SaveAndLoad import LoadImage


data_folder = r'X:\PrcoessedData\PzTzSegment_ZYH\DoctorDraw2'
for case in os.listdir(data_folder):
    case_path = os.path.join(data_folder, case)
    t2_path = os.path.join(case_path, 't2.nii')
    t2_image, t2, _ = LoadImage(t2_path, is_show_info=True)

    # normalize_cg_path = os.path.join(case_path, 'normalize_cg.nii')
    # normalize_cg_image, normalize_cg, _ = LoadImage(normalize_cg_path, is_show_info=True)
    #
    # normalize_prostate_path = os.path.join(case_path, 'normalize_prostate_roi.nii')
    # normalize_prostate_image, normalize_prostate, _ = LoadImage(normalize_prostate_path, is_show_info=True)

    cg_path = os.path.join(case_path, 'cg.nii')
    cg_image, cg, _ = LoadImage(cg_path, is_show_info=True)



    prostate_path = os.path.join(case_path, 'prostate_roi.nii')
    prostate_image, prostate, _ = LoadImage(prostate_path, is_show_info=True)
    print()


