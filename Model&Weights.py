import os

model_name = ['AttenUNet_dice',
              'MSUNet_bce',
              'MultiAttenUNet_bce',
              'MultiUNet_bce',
              'UNet2.5D_bce',
              'UNet_baseline',
              'UNet_baseline_bce+dice',
              'UNet_baseline_dice',
              'WNet_bce']

weights = ['98-1.975697.pt',
           '67-0.751620.pt',
           '45-0.469734.pt',
           '84-0.249598.pt',
           '58-0.229803.pt',
           '78-0.247980.pt',
           '76-2.254344.pt',
           '111-1.942233.pt',
           '75-0.595783.pt']

# print(sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model')))
#
# weights_dict = {}
# for index in weights: