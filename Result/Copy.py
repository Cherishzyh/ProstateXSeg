import shutil
import os

src_root = r'/home/zhangyihong/Documents/Prostate301'
des_root = r'/home/zhangyihong/Documents/MultiVendor/Pred/Data1'
for case in os.listdir(src_root):
    src_path = os.path.join(src_root, '{}/MultiVendor_UNet.nii.gz'.format(case))
    des_folder = os.path.join(src_root, case)
    if not os.path.exists(des_folder):
        os.mkdir(des_folder)
    des_path = os.path.join(des_folder, 'MultiVendor_UNet.nii.gz')
    shutil.copy(src_path, des_path)
    print('copying {}'.format(case))