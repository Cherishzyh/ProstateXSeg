import shutil

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit

from Statistics.Loss import BCEFocalLoss


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='PZ', dtype=np.float32), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='CG', dtype=np.float32), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='U', dtype=np.float32), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='AMSF', dtype=np.float32), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 64

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
    bce = torch.nn.BCELoss()
    # focal = BCEFocalLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):

        train_loss, val_loss = 0., 0.
        train_loss_pz, val_loss_pz = 0., 0.
        train_loss_cg, val_loss_cg = 0., 0.
        train_loss_u, val_loss_u = 0., 0.
        train_loss_asmf, val_loss_asmf = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs)

            loss_pz = bce(preds[:, 0], outputs[0])
            loss_cg = bce(preds[:, 1], outputs[1])
            loss_u = bce(preds[:, 2], outputs[2])
            loss_asmf = bce(preds[:, 3], outputs[3])
            loss = loss_pz + loss_cg + loss_u + loss_asmf

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_pz += loss_pz.item()
            train_loss_cg += loss_cg.item()
            train_loss_u += loss_u.item()
            train_loss_asmf += loss_asmf.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(inputs)

                loss_pz = bce(preds[:, 0], outputs[0])
                loss_cg = bce(preds[:, 1], outputs[1])
                loss_u = bce(preds[:, 2], outputs[2])
                loss_asmf = bce(preds[:, 3], outputs[3])
                loss = loss_pz + loss_cg + loss_u + loss_asmf

                val_loss += loss.item()
                val_loss_pz += loss_pz.item()
                val_loss_cg += loss_cg.item()
                val_loss_u += loss_u.item()
                val_loss_asmf += loss_asmf.item()

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

        print(
            '*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(
                epoch + 1))
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(np.sum(train_dice_pz) / len(train_dice_pz), np.sum(train_dice_cg) / len(train_dice_cg),
                     np.sum(train_dice_U) / len(train_dice_U), np.sum(train_dice_AFMS) / len(train_dice_AFMS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(np.sum(val_dice_pz) / len(val_dice_pz), np.sum(val_dice_cg) / len(val_dice_cg),
                     np.sum(val_dice_U) / len(val_dice_U), np.sum(val_dice_AFMS) / len(val_dice_AFMS)))
        print()
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))
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

    input_shape = (200, 200)
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
            print(sum(outputs))


if __name__ == '__main__':
    from SegModel.ResNet50 import ResUNet4Cls

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'

    model = ResUNet4Cls([2, 3, 3, 3], 3, 5).to(device)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/ResNet50.py'

    Train(model, device, 'ResUNet_cls_0629', py_path)
    # CheckInput()