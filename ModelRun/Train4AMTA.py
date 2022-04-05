import shutil
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit
from CnnTools.T4T.Utility.Loss import FocalLoss

from Statistics.Metric import Dice
from Statistics.Loss import DiceLoss


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 12
    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    CopyFile(net_path, os.path.join(model_folder, 'model.py'), is_replace=True)


    ClearGraphPath(model_folder)

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    train_df = pd.read_csv(os.path.join(data_root, 'train_name.csv'))
    train_list = train_df.values.tolist()[0]
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    model = model.to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 都需要做编码的label，真nice
    focalloss = FocalLoss()
    diceloss = DiceLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_dice, val_dice = [], []
        train_dice_pz, val_dice_pz = [], []
        train_dice_cg, val_dice_cg = [], []
        train_dice_U, val_dice_U = [], []
        train_dice_AFMS, val_dice_AFMS = [], []

        train_loss, val_loss = 0., 0.
        train_loss1, val_loss1 = 0., 0.
        train_loss2, val_loss2 = 0., 0.
        train_loss3, val_loss3 = 0., 0.
        train_loss4, val_loss4 = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_onelabel = torch.argmax(outputs, dim=1, keepdim=True)
            outputs_onelabel = torch.clip(outputs_onelabel, min=0, max=1)

            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs.int(), device)

            outputs_onelabel = MoveTensorsToDevice(outputs_onelabel.int(), device)

            pro, region = model(inputs)

            # train dice
            train_dice.append(Dice(pro.cpu().data.numpy(), outputs_onelabel.cpu().data.numpy()))
            train_dice_pz.append(Dice(region.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
            train_dice_cg.append(Dice(region.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
            train_dice_U.append(Dice(region.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
            train_dice_AFMS.append(Dice(region.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

            loss1 = focalloss(pro, outputs_onelabel)*batch_size
            loss2 = diceloss(pro, outputs_onelabel)
            loss3 = focalloss(region, outputs)*batch_size
            loss4 = diceloss(region, outputs)
            loss = loss1 + loss2 + loss3 + loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += loss4.item()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                outputs_onelabel = torch.argmax(outputs, dim=1, keepdim=True)
                outputs_onelabel = torch.clip(outputs_onelabel, min=0, max=1)

                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs.int(), device)
                outputs_onelabel = MoveTensorsToDevice(outputs_onelabel.int(), device)

                pro, region = model(inputs)

                # train dice
                val_dice.append(Dice(pro.cpu().data.numpy(), outputs_onelabel.cpu().data.numpy()))
                val_dice_pz.append(Dice(region.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
                val_dice_cg.append(Dice(region.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
                val_dice_U.append(Dice(region.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
                val_dice_AFMS.append(Dice(region.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

                loss1 = focalloss(pro, outputs_onelabel) * batch_size
                loss2 = diceloss(pro, outputs_onelabel)
                loss3 = focalloss(region, outputs) * batch_size
                loss4 = diceloss(region, outputs)
                loss = loss1 + loss2 + loss3 + loss4

                val_loss1 += loss1.item()
                val_loss2 += loss2.item()
                val_loss3 += loss3.item()
                val_loss4 += loss4.item()
                val_loss += loss.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_focal_loss_pro': train_loss1 / train_batches,
                            'train_focal_loss_reg': train_loss3 / train_batches,
                            'train_dice_loss_pro': train_loss2 / train_batches,
                            'train_dice_loss_reg': train_loss4 / train_batches,
                            'val_focal_loss_pro': val_loss1 / val_batches,
                            'val_focal_loss_reg': val_loss3 / val_batches,
                            'val_dice_loss_pro': val_loss2 / val_batches,
                            'val_dice_loss_reg': val_loss4 / val_batches}, epoch + 1)

        writer.add_scalars('Total Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)

        writer.add_scalars('Dice',
                           {'train_dice_pro': sum(train_dice) / train_batches,
                            'train_dice_pz': sum(train_dice_pz) / train_batches,
                            'train_dice_cg': sum(train_dice_cg) / train_batches,
                            'train_dice_u': sum(train_dice_U) / train_batches,
                            'train_dice_amsf': sum(train_dice_AFMS) / train_batches,
                            'val_dice_pro': sum(val_dice) / val_batches,
                            'val_dice_pz': sum(val_dice_pz) / val_batches,
                            'val_dice_cg': sum(val_dice_cg) / val_batches,
                            'val_dice_u': sum(val_dice_U) / val_batches,
                            'val_dice_amsf': sum(val_dice_AFMS) / val_batches}, epoch + 1)

        print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
        print('    dice : {:.3f},     dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(sum(train_dice) / train_batches, sum(train_dice_pz) / train_batches, sum(train_dice_cg) / train_batches,
                     sum(train_dice_U) / train_batches, sum(train_dice_AFMS) / train_batches))
        print('val-dice : {:.3f}, val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(sum(val_dice) / val_batches, sum(val_dice_pz) / val_batches, sum(val_dice_cg) / val_batches,
                     sum(val_dice_U) / val_batches, sum(val_dice_AFMS) / val_batches))
        print('')
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


if __name__ == '__main__':
    from SegModel.AMTANet import AMTA_Net

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'

    model = AMTA_Net(3, 1, 5)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/AMTANet.py'

    Train(model, device, 'AMTANet_0623_25D', py_path)
    # CheckInput()