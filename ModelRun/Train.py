import copy
import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from MeDIT.Augment import *
from MeDIT.Others import IterateCase
from MeDIT.Others import MakeFolder, CopyFile
from MeDIT.Visualization import FlattenImages

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit
from T4T.Utility.Loss import FocalLoss

from Statistics.Metric import Dice
from Statistics.Loss import WeightedDiceLoss


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2', shape=input_shape))
    data.AddOne(Image2D(data_root + '/BG', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/CZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/PZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/DPU', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/ASF', shape=input_shape, is_roi=True), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)
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

    spliter = DataSpliter()
    cv_generator = spliter.SplitCV(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/all_train_name.csv', store_root=model_folder)
    for cv_index, (sub_train, sub_val) in enumerate(cv_generator):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

        model = model.to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ce_loss = torch.nn.CrossEntropyLoss()
        dice_loss = WeightedDiceLoss(ignore_index=[0])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.
            train_loss1, val_loss1 = 0., 0.
            train_loss2, val_loss2 = 0., 0.

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(torch.cat(outputs, dim=1), device)

                preds = model(inputs)

                pred_softmax = torch.softmax(preds, dim=1)

                loss1 = ce_loss(preds, torch.argmax(outputs, dim=1))
                loss2 = dice_loss(pred_softmax, outputs)
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
                    outputs = MoveTensorsToDevice(torch.cat(outputs, dim=1), device)

                    preds = model(inputs)

                    pred_softmax = torch.softmax(preds, dim=1)

                    loss1 = ce_loss(preds, torch.argmax(outputs, dim=1))
                    loss2 = dice_loss(pred_softmax, outputs)
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
            writer.add_scalars('Crossentropy Loss',
                               {'train_loss': train_loss1 / train_batches,
                                'val_loss': val_loss1 / val_batches}, epoch + 1)
            writer.add_scalars('Dice Loss',
                               {'train_loss': train_loss2 / train_batches,
                                'val_loss': val_loss2 / val_batches}, epoch + 1)

            print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
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

    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 32
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

    spliter = DataSpliter()
    cv_generator = spliter.SplitCV(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/all_train_name.csv', store_root=model_folder)
    for cv_index, (sub_train, sub_val) in enumerate(cv_generator):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

        model = model.to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ce_loss = torch.nn.CrossEntropyLoss()
        dice_loss = WeightedDiceLoss(ignore_index=[0])
        mse_loss = torch.nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.
            train_loss1, val_loss1 = 0., 0.
            train_loss2, val_loss2 = 0., 0.

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(torch.cat(outputs, dim=1), device)
                outputs_2Cl = torch.cat([outputs[:, :2], torch.sum(outputs[:, 2:], dim=1, keepdim=True)], dim=1)

                preds = model(inputs, epoch)

                pred_1, pred_2 = preds[0], preds[1]
                pred_1_softmax, pred_2_softmax = torch.softmax(pred_1, dim=1), torch.softmax(pred_2, dim=1)

                loss1 = ce_loss(pred_1, torch.argmax(outputs_2Cl, dim=1)) + ce_loss(pred_2, torch.argmax(outputs, dim=1))
                loss2 = dice_loss(pred_1_softmax, outputs_2Cl) + dice_loss(pred_2_softmax, outputs)
                loss = loss1 + loss2
                # if epoch > 5:
                #     pred_2_2Cl = torch.cat([pred_2_softmax[:, :2], torch.sum(pred_2_softmax[:, 2:], dim=1, keepdim=True)], dim=1)
                #     loss += mse_loss(pred_1_softmax, pred_2_2Cl)

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
                    outputs = MoveTensorsToDevice(torch.cat(outputs, dim=1), device)
                    outputs_2Cl = torch.cat([outputs[:, :2], torch.sum(outputs[:, 2:], dim=1, keepdim=True)], dim=1)

                    preds = model(inputs, epoch)

                    pred_1, pred_2 = preds[0], preds[1]
                    pred_1_softmax, pred_2_softmax = torch.softmax(pred_1, dim=1), torch.softmax(pred_2, dim=1)

                    loss1 = ce_loss(pred_1, torch.argmax(outputs_2Cl, dim=1)) + ce_loss(pred_2,
                                                                                        torch.argmax(outputs, dim=1))
                    loss2 = dice_loss(pred_1_softmax, outputs_2Cl) + dice_loss(pred_2_softmax, outputs)
                    loss = loss1 + loss2
                    # if epoch > 5:
                    #     pred_2_2Cl = torch.cat(
                    #         [pred_2_softmax[:, :2], torch.sum(pred_2_softmax[:, 2:], dim=1, keepdim=True)], dim=1)
                    #     loss += mse_loss(pred_1_softmax, pred_2_2Cl)

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
            writer.add_scalars('Crossentropy Loss',
                               {'train_loss': train_loss1 / train_batches,
                                'val_loss': val_loss1 / val_batches}, epoch + 1)
            writer.add_scalars('Dice Loss',
                               {'train_loss': train_loss2 / train_batches,
                                'val_loss': val_loss2 / val_batches}, epoch + 1)

            print('*************************************** Epoch {} | (◕ᴗ◕✿) ***************************************'.format(epoch + 1))
            print('loss: {:.3f}, val-loss: {:.3f}'.format(train_loss / train_batches, val_loss / val_batches))

            scheduler.step(val_loss)
            early_stopping(val_loss, model, (epoch + 1, val_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.flush()
        writer.close()


def TestW(model, device, model_name, data_type):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (192, 192)
    batch_size = 32

    df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', '{}_name.csv'.format(data_type)))
    sub_list = df.values.tolist()[0]

    loader, batches = _GetLoader(sub_list, None, input_shape, batch_size, True)

    model = model.to(device)
    one_fold_weights_list = [one for one in IterateCase(os.path.join(model_root, model_name), only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
    one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
    weights_path = one_fold_weights_list[-1]
    print(weights_path.name)
    model.load_state_dict(torch.load(str(weights_path)))

    model.eval()
    pred_1_list, pred_2_list = [], []
    t2_list, label_list = [], []
    with torch.no_grad():
        for ind, (inputs, outputs) in enumerate(loader):
            inputs = MoveTensorsToDevice(inputs, device)
            # outputs = MoveTensorsToDevice(torch.cat(outputs, dim=1), device)
            # outputs_2Cl = torch.cat([outputs[:, :2], torch.sum(outputs[:, 2:], dim=1, keepdim=True)], dim=1)

            preds = model(inputs)

            pred_1, pred_2 = preds[0], preds[1]
            pred_1_softmax, pred_2_softmax = torch.softmax(pred_1, dim=1), torch.softmax(pred_2, dim=1)
            pred_1_list.extend(list(torch.argmax(pred_1_softmax, dim=1).cpu().detach().numpy()))
            pred_2_list.extend(list(torch.argmax(pred_2_softmax, dim=1).cpu().detach().numpy()))
            t2_list.extend(list(torch.squeeze(inputs).cpu().detach().numpy()))
            label_list.extend(list(torch.argmax(torch.cat(outputs, dim=1), dim=1).cpu().detach().numpy()))
            # print()
        save_path = os.path.join(os.path.join(model_root, model_name), 'Result')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, '{}_t2.npy'.format(data_type)), np.array(t2_list))
        np.save(os.path.join(save_path, '{}_label.npy'.format(data_type)), np.array(label_list))
        np.save(os.path.join(save_path, '{}_pred1.npy'.format(data_type)), np.array(pred_1_list))
        np.save(os.path.join(save_path, '{}_pred2.npy'.format(data_type)), np.array(pred_2_list))


if __name__ == '__main__':
    from SegModel.SuccessfulWNet import WNet2_5D, WNet2_5D_channelcombine, WNet2_5D_channelcombine_share, UNet

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm'

    # model = WNet2_5D_channelcombine_share(3, 3, 5)
    model = UNet(3, 5)
    py_path = r'/home/zhangyihong/SSHProject/ProstateXSeg/SegModel/SuccessfulWNet.py'

    model_name = 'UNet_20220307'

    Train(model, device, model_name, py_path)

    # TrainW(model, device, model_name, py_path)
    # TestW(model, device, model_name, 'train')
    # TestW(model, device, model_name, 'val')
    # TestW(model, device, model_name, 'test')

    # CheckInput()
    # train_list = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/train_case_name.csv').values[0].tolist()
    # val_list = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/val_case_name.csv').values[0].tolist()
    # train_list.extend(val_list)
    # new_df = pd.DataFrame(sorted(train_list)).T
    # new_df.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/all_train_case_name.csv', index=False)
