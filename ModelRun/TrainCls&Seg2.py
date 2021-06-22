import shutil

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit

from Statistics.Loss import DiceLoss
from Statistics.Metric import Dice

def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='PZ'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='CG'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='U'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='AMSF'), is_input=False)
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 48

    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    ClearGraphPath(model_folder)

    CopyFile(net_path, os.path.join(model_folder, 'model.py'), is_replace=True)

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
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    seg_dice = DiceLoss()
    seg_ce = torch.nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):

        train_loss, val_loss = 0., 0.
        train_loss_pz, val_loss_pz = 0., 0.
        train_loss_cg, val_loss_cg = 0., 0.
        train_loss_u, val_loss_u = 0., 0.
        train_loss_asmf, val_loss_asmf = 0., 0.
        train_loss_dice, val_loss_dice = 0., 0.
        train_loss_ce, val_loss_ce = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_nocoding = torch.argmax(outputs[4], dim=1)
            outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            cls, seg = model(inputs)

            loss_pz = cls_criterion(cls[:, 0], outputs[0])
            loss_cg = cls_criterion(cls[:, 1], outputs[1])
            loss_u = cls_criterion(cls[:, 2], outputs[2])
            loss_asmf = cls_criterion(cls[:, 3], outputs[3])

            seg_pred = torch.softmax(seg, dim=1)
            seg_pred_cg = (seg_pred[:, 1, ...].transpose(0, 1).transpose(1, 2) * cls[:, 0]).transpose(1, 2).transpose(0, 1)
            seg_pred_pz = (seg_pred[:, 2, ...].transpose(0, 1).transpose(1, 2) * cls[:, 1]).transpose(1, 2).transpose(0, 1)
            seg_pred_u = (seg_pred[:, 3, ...].transpose(0, 1).transpose(1, 2) * cls[:, 2]).transpose(1, 2).transpose(0, 1)
            seg_pred_as = (seg_pred[:, 4, ...].transpose(0, 1).transpose(1, 2) * cls[:, 3]).transpose(1, 2).transpose(0, 1)
            new_seg = torch.stack((seg_pred[:, 0, ...], seg_pred_cg, seg_pred_pz, seg_pred_u, seg_pred_as), dim=1)
            seg_pred_new = torch.softmax(new_seg, dim=1)

            loss_dice = seg_dice(seg_pred_new, outputs[4])
            loss_ce = seg_ce(seg_pred_new.log(), outputs_nocoding)
            loss = loss_pz + loss_cg + loss_u + loss_asmf + loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_pz += loss_pz.item()
            train_loss_cg += loss_cg.item()
            train_loss_u += loss_u.item()
            train_loss_asmf += loss_asmf.item()
            train_loss_dice += loss_dice.item()
            train_loss_ce += loss_ce.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                outputs_nocoding = torch.argmax(outputs[4], dim=1)
                outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                cls, seg = model(inputs)

                loss_pz = cls_criterion(cls[:, 0], outputs[0])
                loss_cg = cls_criterion(cls[:, 1], outputs[1])
                loss_u = cls_criterion(cls[:, 2], outputs[2])
                loss_asmf = cls_criterion(cls[:, 3], outputs[3])

                seg_pred = torch.softmax(seg, dim=1)
                seg_pred_cg = (seg_pred[:, 1, ...].transpose(0, 1).transpose(1, 2) * cls[:, 0]).transpose(1, 2).transpose(0, 1)
                seg_pred_pz = (seg_pred[:, 2, ...].transpose(0, 1).transpose(1, 2) * cls[:, 1]).transpose(1, 2).transpose(0, 1)
                seg_pred_u = (seg_pred[:, 3, ...].transpose(0, 1).transpose(1, 2) * cls[:, 2]).transpose(1, 2).transpose(0, 1)
                seg_pred_as = (seg_pred[:, 4, ...].transpose(0, 1).transpose(1, 2) * cls[:, 3]).transpose(1, 2).transpose(0, 1)
                new_seg = torch.stack((seg_pred[:, 0, ...], seg_pred_cg, seg_pred_pz, seg_pred_u, seg_pred_as), dim=1)
                seg_pred_new = torch.softmax(new_seg, dim=1)

                loss_dice = seg_dice(seg_pred_new, outputs[4])
                loss_ce = seg_ce(seg_pred_new.log(), outputs_nocoding)
                loss = loss_pz + loss_cg + loss_u + loss_asmf + loss_ce + loss_dice

                val_loss += loss.item()
                val_loss_pz += loss_pz.item()
                val_loss_cg += loss_cg.item()
                val_loss_u += loss_u.item()
                val_loss_asmf += loss_asmf.item()
                val_loss_dice += loss_dice.item()
                val_loss_ce += loss_ce.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_pz',
                           {'train_loss': train_loss_pz / train_batches, 'val_loss': val_loss_pz / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_cg',
                           {'train_loss': train_loss_cg / train_batches, 'val_loss': val_loss_cg / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_u',
                           {'train_loss': train_loss_u / train_batches, 'val_loss': val_loss_u / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_asmf',
                           {'train_loss': train_loss_asmf / train_batches, 'val_loss': val_loss_asmf / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_dice',
                           {'train_loss': train_loss_dice / train_batches, 'val_loss': val_loss_dice / val_batches},
                           epoch + 1)
        writer.add_scalars('Loss_ce',
                           {'train_loss': train_loss_ce / train_batches, 'val_loss': val_loss_ce / val_batches},
                           epoch + 1)


        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def CheckInput():
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 24

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

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)

    for epoch in range(total_epoch):
        for ind, (inputs, outputs) in enumerate(train_loader):
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)
            print()


if __name__ == '__main__':
    from SegModel.ResNet50 import ModelRun

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'

    model = ModelRun(1, 4, res_num=34, seg=True).to(device)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/ResNet50.py'

    Train(model, device, 'ResUNet34_0616_multiloss', py_path)
    # CheckInput()