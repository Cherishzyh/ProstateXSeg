import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit
from CnnTools.T4T.Utility.Loss import FocalLoss

from Statistics.Loss import DistLoss
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
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 36
    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    ClearGraphPath(model_folder)
    if net_path.endswith('.py'):
        CopyFile(net_path, os.path.join(model_folder, 'model.py'), is_replace=True)
    else:
        shutil.copytree(net_path, os.path.join(model_folder, 'model'))

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        # BiasTransform.name: {'center': ['uniform', -1., 1., 2],
        #                      'drop_ratio': ['uniform', 0., 1.]},
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
    focalloss = FocalLoss()
    distloss = DistLoss()

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

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            inputs = MoveTensorsToDevice(inputs, device)
            label = MoveTensorsToDevice(outputs[0].int(), device)
            dismap = MoveTensorsToDevice(outputs[1], device)

            preds = model(inputs)
            if isinstance(preds, tuple):
                preds = preds[-1]

            train_dice.append(Dice(preds.cpu().data.numpy(), label.cpu().data.numpy()))
            train_dice_pz.append(Dice(preds.cpu().data.numpy()[:, 1], label.cpu().data.numpy()[:, 1]))
            train_dice_cg.append(Dice(preds.cpu().data.numpy()[:, 2], label.cpu().data.numpy()[:, 2]))
            train_dice_U.append(Dice(preds.cpu().data.numpy()[:, 3], label.cpu().data.numpy()[:, 3]))
            train_dice_AFMS.append(Dice(preds.cpu().data.numpy()[:, 4], label.cpu().data.numpy()[:, 4]))

            loss1 = focalloss(preds, label)*batch_size
            loss2 = distloss(dismap, preds)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                label = MoveTensorsToDevice(outputs[0].int(), device)
                dismap = MoveTensorsToDevice(outputs[1], device)

                preds = model(inputs)
                if isinstance(preds, tuple):
                    preds = preds[-1]

                val_dice.append(Dice(preds.cpu().data.numpy(), label.cpu().data.numpy()))
                val_dice_pz.append(Dice(preds.cpu().data.numpy()[:, 1], label.cpu().data.numpy()[:, 1]))
                val_dice_cg.append(Dice(preds.cpu().data.numpy()[:, 2], label.cpu().data.numpy()[:, 2]))
                val_dice_U.append(Dice(preds.cpu().data.numpy()[:, 3], label.cpu().data.numpy()[:, 3]))
                val_dice_AFMS.append(Dice(preds.cpu().data.numpy()[:, 4], label.cpu().data.numpy()[:, 4]))

                loss1 = focalloss(preds, label) * batch_size
                loss2 = distloss(dismap, preds)
                loss = loss1 + loss2

                val_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)
        writer.add_scalars('Focal Loss Loss',
                           {'train_loss': train_loss1 / train_batches,
                            'val_loss': val_loss1 / val_batches}, epoch + 1)
        writer.add_scalars('Dist Loss',
                           {'train_loss': train_loss2 / train_batches,
                            'val_loss': val_loss2 / val_batches}, epoch + 1)
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
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(np.sum(train_dice_pz) / len(train_dice_pz), np.sum(train_dice_cg) / len(train_dice_cg),
                     np.sum(train_dice_U) / len(train_dice_U), np.sum(train_dice_AFMS) / len(train_dice_AFMS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(np.sum(val_dice_pz) / len(val_dice_pz), np.sum(val_dice_cg) / len(val_dice_cg),
                     np.sum(val_dice_U) / len(val_dice_U), np.sum(val_dice_AFMS) / len(val_dice_AFMS)))
        print()
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def Train1(model, device, model_name, net_path):

    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 40
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
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = DiceLoss()
    criterion3 = torch.nn.BCEWithLogitsLoss()

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
        train_dice_pro, val_dice_pro = [], []

        train_loss, val_loss = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_nocoding = torch.argmax(outputs, dim=1)
            outputs_nocoding_pro = torch.clamp(outputs_nocoding, min=0, max=1)

            inputs = MoveTensorsToDevice(inputs, device)
            outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
            outputs_nocoding_pro = MoveTensorsToDevice(outputs_nocoding_pro, device)
            outputs = MoveTensorsToDevice(outputs.int(), device)


            pred_pro, preds = model(inputs)

            sigmoid_pred_pro = torch.sigmoid(pred_pro)
            softmax_preds = F.softmax(preds, dim=1)

            train_dice.append(Dice(softmax_preds.cpu().data.numpy(), outputs.cpu().data.numpy()))
            train_dice_pz.append(Dice(softmax_preds.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
            train_dice_cg.append(Dice(softmax_preds.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
            train_dice_U.append(Dice(softmax_preds.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
            train_dice_AFMS.append(Dice(softmax_preds.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

            train_dice_pro.append(Dice(torch.squeeze(sigmoid_pred_pro).cpu().data.numpy(), outputs_nocoding_pro.cpu().data.numpy()))

            loss1 = criterion1(preds, outputs_nocoding)
            loss2 = criterion2(softmax_preds, outputs)
            loss3 = criterion3(torch.squeeze(pred_pro, dim=1), outputs_nocoding_pro.float())
            loss4 = criterion2(sigmoid_pred_pro, outputs_nocoding_pro)
            loss = loss1 + loss2 + loss3 + loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):

                outputs_nocoding = torch.argmax(outputs, dim=1)
                outputs_nocoding_pro = torch.clamp(outputs_nocoding, min=0, max=1)

                inputs = MoveTensorsToDevice(inputs, device)
                outputs_nocoding = MoveTensorsToDevice(outputs_nocoding, device)
                outputs_nocoding_pro = MoveTensorsToDevice(outputs_nocoding_pro, device)
                outputs = MoveTensorsToDevice(outputs.int(), device)

                pred_pro, preds = model(inputs)

                sigmoid_pred_pro = torch.sigmoid(pred_pro)
                softmax_preds = F.softmax(preds, dim=1)

                val_dice.append(Dice(softmax_preds.cpu().data.numpy(), outputs.cpu().data.numpy()))
                val_dice_pz.append(Dice(softmax_preds.cpu().data.numpy()[:, 1], outputs.cpu().data.numpy()[:, 1]))
                val_dice_cg.append(Dice(softmax_preds.cpu().data.numpy()[:, 2], outputs.cpu().data.numpy()[:, 2]))
                val_dice_U.append(Dice(softmax_preds.cpu().data.numpy()[:, 3], outputs.cpu().data.numpy()[:, 3]))
                val_dice_AFMS.append(
                    Dice(softmax_preds.cpu().data.numpy()[:, 4], outputs.cpu().data.numpy()[:, 4]))

                val_dice_pro.append(
                    Dice(torch.squeeze(sigmoid_pred_pro).cpu().data.numpy(), outputs_nocoding_pro.cpu().data.numpy()))

                loss1 = criterion1(preds, outputs_nocoding)
                loss2 = criterion2(softmax_preds, outputs)
                loss3 = criterion3(torch.squeeze(pred_pro, dim=1), outputs_nocoding_pro.float())
                loss4 = criterion2(sigmoid_pred_pro, outputs_nocoding_pro)
                loss = loss1 + loss2 + loss3 + loss4

                val_loss += loss.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches}, epoch + 1)

        writer.add_scalars('Dice Pro',
                           {'train_dice': np.sum(train_dice_pro) / len(train_dice_pro),
                            'val_dice': np.sum(val_dice_pro) / len(val_dice_pro)}, epoch + 1)
        writer.add_scalars('Dice',
                           {'train_dice': np.sum(train_dice) / len(train_dice),
                            'val_dice': np.sum(val_dice) / len(val_dice)}, epoch + 1)

        print()
        print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
        print()
        print('dice pro: {:3f}, val-dice pro: {:3f}'.format(np.sum(train_dice_pro) / len(train_dice_pro), np.sum(val_dice_pro) / len(val_dice_pro)))
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(np.sum(train_dice_pz) / len(train_dice_pz), np.sum(train_dice_cg) / len(train_dice_cg),
                     np.sum(train_dice_U) / len(train_dice_U), np.sum(train_dice_AFMS) / len(train_dice_AFMS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(np.sum(val_dice_pz) / len(val_dice_pz), np.sum(val_dice_cg) / len(val_dice_cg),
                     np.sum(val_dice_U) / len(val_dice_U), np.sum(val_dice_AFMS) / len(val_dice_AFMS)))
        print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def TrainW(model, device, model_name, net_path):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 20
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
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = DiceLoss()

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

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):

            outputs_nocoding_5 = torch.argmax(outputs, dim=1)
            outputs_nocoding_3 = torch.clamp_max(outputs_nocoding_5, 2)

            outputs_5 = outputs
            outputs_3 = torch.cat([outputs[:, 0:2, ...], torch.sum(outputs[:, 2:, ...], dim=1, keepdim=True)], dim=1)

            inputs = MoveTensorsToDevice(inputs, device)

            outputs_nocoding_5 = MoveTensorsToDevice(outputs_nocoding_5, device)
            outputs_5 = MoveTensorsToDevice(outputs_5.int(), device)

            outputs_nocoding_3 = MoveTensorsToDevice(outputs_nocoding_3, device)
            outputs_3 = MoveTensorsToDevice(outputs_3.int(), device)

            #  preds1: bg, pz, other    |     preds2: bg, pz, cg, u, amsf
            preds1, preds2 = model(inputs)

            # Crossentropy Loss: preds: logits(没有做softmax)
            #                    labels: 没有做编码
            # Dice Loss: preds: 要做softmax
            #            labels: 要做编码，格式和preds相同

            softmax_preds1 = F.softmax(preds1, dim=1)
            softmax_preds2 = F.softmax(preds2, dim=1)

            train_dice.append(Dice(softmax_preds2.cpu().data.numpy(), outputs_5.cpu().data.numpy()))
            train_dice_pz.append(Dice(softmax_preds2.cpu().data.numpy()[:, 1], outputs_5.cpu().data.numpy()[:, 1]))
            train_dice_cg.append(Dice(softmax_preds2.cpu().data.numpy()[:, 2], outputs_5.cpu().data.numpy()[:, 2]))
            train_dice_U.append(Dice(softmax_preds2.cpu().data.numpy()[:, 3], outputs_5.cpu().data.numpy()[:, 3]))
            train_dice_AFMS.append(Dice(softmax_preds2.cpu().data.numpy()[:, 4], outputs_5.cpu().data.numpy()[:, 4]))

            # loss1: loss of pred1
            loss1 = criterion1(preds1, outputs_nocoding_3) + criterion2(softmax_preds1, outputs_3)
            # loss2: loss of pred2
            loss2 = criterion1(preds2, outputs_nocoding_5) + criterion2(softmax_preds2, outputs_5)
            # loss3: loss between pred1 & pred2

            if epoch >= 5:
                label_3 = torch.argmax(preds1, dim=1)
                preds2_3 = torch.cat([preds2[:, 0:2, ...], torch.sum(preds2[:, 2:, ...], dim=1, keepdim=True)], dim=1)

                loss3 = criterion1(preds2_3, label_3)

                loss = loss1 + loss2 + loss3
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss += loss.item()
            else:

                loss = loss1 + loss2
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):

                outputs_nocoding_5 = torch.argmax(outputs, dim=1)
                outputs_nocoding_3 = torch.clamp_max(outputs_nocoding_5, 2)

                outputs_5 = outputs
                outputs_3 = torch.cat([outputs[:, 0:2, ...], torch.sum(outputs[:, 2:, ...], dim=1, keepdim=True)],
                                      dim=1)

                inputs = MoveTensorsToDevice(inputs, device)

                outputs_nocoding_5 = MoveTensorsToDevice(outputs_nocoding_5, device)
                outputs_5 = MoveTensorsToDevice(outputs_5.int(), device)

                outputs_nocoding_3 = MoveTensorsToDevice(outputs_nocoding_3, device)
                outputs_3 = MoveTensorsToDevice(outputs_3.int(), device)

                preds1, preds2 = model(inputs)

                softmax_preds1 = F.softmax(preds1, dim=1)
                softmax_preds2 = F.softmax(preds2, dim=1)

                val_dice.append(Dice(softmax_preds2.cpu().data.numpy(), outputs_5.cpu().data.numpy()))
                val_dice_pz.append(Dice(softmax_preds2.cpu().data.numpy()[:, 1], outputs_5.cpu().data.numpy()[:, 1]))
                val_dice_cg.append(Dice(softmax_preds2.cpu().data.numpy()[:, 2], outputs_5.cpu().data.numpy()[:, 2]))
                val_dice_U.append(Dice(softmax_preds2.cpu().data.numpy()[:, 3], outputs_5.cpu().data.numpy()[:, 3]))
                val_dice_AFMS.append(Dice(softmax_preds2.cpu().data.numpy()[:, 4], outputs_5.cpu().data.numpy()[:, 4]))

                # loss1: loss of pred1
                loss1 = criterion1(preds1, outputs_nocoding_3) + criterion2(softmax_preds1, outputs_3)
                # loss2: loss of pred2
                loss2 = criterion1(preds2, outputs_nocoding_5) + criterion2(softmax_preds2, outputs_5)
                # loss3: loss between pred1 & pred2

                if epoch >= 5:
                    label_3 = torch.argmax(preds1, dim=1)
                    preds2_3 = torch.cat([preds2[:, 0:2, ...], torch.sum(preds2[:, 2:, ...], dim=1, keepdim=True)],
                                         dim=1)
                    loss3 = criterion1(preds2_3, label_3)

                    loss = loss1 + loss2 + loss3
                    val_loss1 += loss1.item()
                    val_loss2 += loss2.item()
                    val_loss3 += loss3.item()
                    val_loss += loss.item()
                else:
                    loss = loss1 + loss2
                    val_loss1 += loss1.item()
                    val_loss2 += loss2.item()
                    val_loss += loss.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)
        writer.add_scalars('UNet1',
                           {'train_loss': train_loss1 / train_batches,
                            'val_loss': val_loss1 / val_batches}, epoch + 1)
        writer.add_scalars('UNet2',
                           {'train_loss': train_loss2 / train_batches,
                            'val_loss': val_loss2 / val_batches}, epoch + 1)
        writer.add_scalars('Loss3',
                           {'train_loss': train_loss3 / train_batches,
                            'val_loss': val_loss3 / val_batches}, epoch + 1)

        writer.add_scalars('Dice',
                           {'train_dice': np.sum(train_dice) / len(train_dice),
                            'val_dice': np.sum(val_dice) / len(val_dice)}, epoch + 1)

        print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
        print('    dice pz: {:.3f},     dice cg: {:.3f},     dice U: {:.3f},     dice AFMS: {:.3f}'.
              format(np.sum(train_dice_pz) / len(train_dice_pz), np.sum(train_dice_cg) / len(train_dice_cg),
                     np.sum(train_dice_U) / len(train_dice_U), np.sum(train_dice_AFMS) / len(train_dice_AFMS)))
        print('val-dice pz: {:.3f}, val-dice cg: {:.3f}, val-dice U: {:.3f}, val-dice AFMS: {:.3f}'.
              format(np.sum(val_dice_pz) / len(val_dice_pz), np.sum(val_dice_cg) / len(val_dice_cg),
                     np.sum(val_dice_U) / len(val_dice_U), np.sum(val_dice_AFMS) / len(val_dice_AFMS)))
        print()
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
    val_df = pd.read_csv(os.path.join(data_root, 'val_name.csv'))
    val_list = val_df.values.tolist()[0]

    train_loader, train_batches = _GetLoader(train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_list, param_config, input_shape, batch_size, True)

    for epoch in range(total_epoch):
        for ind, (inputs, outputs) in enumerate(train_loader):
            outputs_roi = torch.argmax(outputs, dim=1).numpy()
            for index in range(inputs.shape[0]):
                plt.imshow(inputs[index, 1, ...].numpy(), cmap='gray')
                plt.contour(outputs_roi[index])
                plt.show()


if __name__ == '__main__':
    from SegModel.UNet_Git.unet_model import UNet

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'

    model = UNet(1, 5)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/UNet_Git'

    Train(model, device, 'UNet_0626_focal_dist', py_path)
    # CheckInput()