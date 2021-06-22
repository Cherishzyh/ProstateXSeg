import shutil

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from BasicTool.MeDIT.Statistics import BinaryClassification
from CnnTools.T4T.Utility.Data import *

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


def Test(model, device, model_folder, epoch, data_type='train'):
    torch.autograd.set_detect_anomaly(True)

    test_df = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)))
    test_list = test_df.values.tolist()[0]
    data_loader, data_batches = _GetLoader(test_list, None, (200, 200), 48, False)

    model.to(device)
    model.load_state_dict(torch.load(model_folder + '/{}'.format(epoch)))

    pz_list, cg_list, u_list, asmf_list = [], [], [], []
    pz_label, cg_label, u_label, asmf_label = [], [], [], []
    model.eval()
    with torch.no_grad():
        for ind, (inputs, outputs) in enumerate(data_loader):
            inputs = MoveTensorsToDevice(inputs, device)

            preds = model(inputs)
            preds = torch.sigmoid(preds)

            pz_list.extend(preds[:, 0].cpu().data.numpy().tolist())
            pz_label.extend(outputs[0].numpy().tolist())
            cg_list.extend(preds[:, 1].cpu().data.numpy().tolist())
            cg_label.extend(outputs[1].numpy().tolist())
            u_list.extend(preds[:, 2].cpu().data.numpy().tolist())
            u_label.extend(outputs[2].numpy().tolist())
            asmf_list.extend(preds[:, 3].cpu().data.numpy().tolist())
            asmf_label.extend(outputs[3].numpy().tolist())

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_pz_preds.npy'.format(data_type)), np.array(pz_list))
    np.save(os.path.join(result_folder, '{}_pz_label.npy'.format(data_type)), np.array(pz_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_cg_preds.npy'.format(data_type)), np.array(cg_list))
    np.save(os.path.join(result_folder, '{}_cg_label.npy'.format(data_type)), np.array(cg_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_u_preds.npy'.format(data_type)), np.array(u_list))
    np.save(os.path.join(result_folder, '{}_u_label.npy'.format(data_type)), np.array(u_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_asmf_preds.npy'.format(data_type)), np.array(asmf_list))
    np.save(os.path.join(result_folder, '{}_asmf_label.npy'.format(data_type)), np.array(asmf_label, dtype=np.int))


def TestClsSeg(model, device, model_folder, epoch, data_type='train'):
    torch.autograd.set_detect_anomaly(True)

    df = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)))
    data_list = df.values.tolist()[0]
    data_loader, data_batches = _GetLoader(data_list, None, (192, 192), 48, False)

    model.to(device)
    model.load_state_dict(torch.load(model_folder + '/{}'.format(epoch)))

    pz_list, cg_list, u_list, asmf_list = [], [], [], []
    pz_label, cg_label, u_label, asmf_label = [], [], [], []
    seg_label, seg_pred = [], []

    model.eval()
    with torch.no_grad():
        for ind, (inputs, outputs) in enumerate(data_loader):
            inputs = MoveTensorsToDevice(inputs, device)

            preds, seg = model(inputs)

            preds = torch.sigmoid(preds)
            pz_list.extend(preds[:, 0].cpu().data.numpy().tolist())
            pz_label.extend(outputs[0].numpy().tolist())
            cg_list.extend(preds[:, 1].cpu().data.numpy().tolist())
            cg_label.extend(outputs[1].numpy().tolist())
            u_list.extend(preds[:, 2].cpu().data.numpy().tolist())
            u_label.extend(outputs[2].numpy().tolist())
            asmf_list.extend(preds[:, 3].cpu().data.numpy().tolist())
            asmf_label.extend(outputs[3].numpy().tolist())

            seg = torch.softmax(seg, dim=1)
            seg_pred.extend(list(seg.cpu().data.numpy()))
            seg_label.extend(list(outputs[4].numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_pz_preds.npy'.format(data_type)), np.array(pz_list))
    np.save(os.path.join(result_folder, '{}_pz_label.npy'.format(data_type)), np.array(pz_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_cg_preds.npy'.format(data_type)), np.array(cg_list))
    np.save(os.path.join(result_folder, '{}_cg_label.npy'.format(data_type)), np.array(cg_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_u_preds.npy'.format(data_type)), np.array(u_list))
    np.save(os.path.join(result_folder, '{}_u_label.npy'.format(data_type)), np.array(u_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_asmf_preds.npy'.format(data_type)), np.array(asmf_list))
    np.save(os.path.join(result_folder, '{}_asmf_label.npy'.format(data_type)), np.array(asmf_label, dtype=np.int))
    np.save(os.path.join(result_folder, '{}_seg_preds.npy'.format(data_type)), np.array(seg_pred))
    np.save(os.path.join(result_folder, '{}_seg_label.npy'.format(data_type)), np.array(seg_label, dtype=np.int))


def ComputeAUC(result_folder, data_type='train'):
    bc = BinaryClassification()

    pz_preds = np.load(os.path.join(result_folder, '{}_pz_preds.npy'.format(data_type))).tolist()
    pz_label = np.load(os.path.join(result_folder, '{}_pz_label.npy'.format(data_type))).tolist()
    cg_preds = np.load(os.path.join(result_folder, '{}_cg_preds.npy'.format(data_type))).tolist()
    cg_label = np.load(os.path.join(result_folder, '{}_cg_label.npy'.format(data_type))).tolist()
    u_preds = np.load(os.path.join(result_folder, '{}_u_preds.npy'.format(data_type))).tolist()
    u_label = np.load(os.path.join(result_folder, '{}_u_label.npy'.format(data_type))).tolist()
    asmf_preds = np.load(os.path.join(result_folder, '{}_asmf_preds.npy'.format(data_type))).tolist()
    asmf_label = np.load(os.path.join(result_folder, '{}_asmf_label.npy'.format(data_type))).tolist()

    print('**************************** PZ ********************************')
    bc.Run(pz_preds, pz_label)
    print('**************************** CG ********************************')
    bc.Run(cg_preds, cg_label)
    print('**************************** U  ********************************')
    bc.Run(u_preds, u_label)
    print('****************************ASMF********************************')
    bc.Run(asmf_preds, asmf_label)


def ComputeClassAUC(result_folder, data_type='pz'):
    train_preds = np.load(os.path.join(result_folder, 'train_{}_preds.npy'.format(data_type))).tolist()
    train_label = np.load(os.path.join(result_folder, 'train_{}_label.npy'.format(data_type))).tolist()
    val_preds = np.load(os.path.join(result_folder, 'val_{}_preds.npy'.format(data_type))).tolist()
    val_label = np.load(os.path.join(result_folder, 'val_{}_label.npy'.format(data_type))).tolist()
    test_preds = np.load(os.path.join(result_folder, 'test_{}_preds.npy'.format(data_type))).tolist()
    test_label = np.load(os.path.join(result_folder, 'test_{}_label.npy'.format(data_type))).tolist()

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(train_label, train_preds)
    auc = roc_auc_score(train_label, train_preds)
    plt.plot(fpn, sen, label='train {}: {:.3f}'.format(data_type, auc))

    fpn, sen, the = roc_curve(val_label, val_preds)
    auc = roc_auc_score(val_label, val_preds)
    plt.plot(fpn, sen, label='val {}: {:.3f}'.format(data_type, auc))

    fpn, sen, the = roc_curve(test_label, test_preds)
    auc = roc_auc_score(test_label, test_preds)
    plt.plot(fpn, sen, label='test {}: {:.3f}'.format(data_type, auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def ComputeDice(result_folder):
    train_preds = np.load(os.path.join(result_folder, '{}_seg_preds.npy'.format('train')))
    train_label = np.load(os.path.join(result_folder, '{}_seg_label.npy'.format('train')))
    val_preds = np.load(os.path.join(result_folder, '{}_seg_preds.npy'.format('val')))
    val_label = np.load(os.path.join(result_folder, '{}_seg_label.npy'.format('val')))
    test_preds = np.load(os.path.join(result_folder, '{}_seg_preds.npy'.format('test')))
    test_label = np.load(os.path.join(result_folder, '{}_seg_label.npy'.format('test')))

    print('**************************** Train ********************************')
    print('BG: {:.3f}, PZ: {:.3f}, CG: {:.3f}, U: {:.3f}, ASMF: {:.3f}'.format(
        Dice(train_preds[:, 0], train_label[:, 0]),
        Dice(train_preds[:, 1], train_label[:, 1]),
        Dice(train_preds[:, 2], train_label[:, 2]),
        Dice(train_preds[:, 3], train_label[:, 3]),
        Dice(train_preds[:, 4], train_label[:, 4]),
    ))
    print('**************************** Val ********************************')
    print('BG: {:.3f}, PZ: {:.3f}, CG: {:.3f}, U: {:.3f}, ASMF: {:.3f}'.format(
        Dice(val_preds[:, 0], val_label[:, 0]),
        Dice(val_preds[:, 1], val_label[:, 1]),
        Dice(val_preds[:, 2], val_label[:, 2]),
        Dice(val_preds[:, 3], val_label[:, 3]),
        Dice(val_preds[:, 4], val_label[:, 4]),
    ))
    print('**************************** Test  ********************************')
    print('BG: {:.3f}, PZ: {:.3f}, CG: {:.3f}, U: {:.3f}, AFMS: {:.3f}'.format(
        Dice(test_preds[:, 0], test_label[:, 0]),
        Dice(test_preds[:, 1], test_label[:, 1]),
        Dice(test_preds[:, 2], test_label[:, 2]),
        Dice(test_preds[:, 3], test_label[:, 3]),
        Dice(test_preds[:, 4], test_label[:, 4]),
    ))


if __name__ == '__main__':
    from SegModel.ResNet50 import ModelRun

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'

    model = ModelRun(1, 4, res_num=34, seg=True).to(device)

    # model_folder = os.path.join(model_root, 'ResNet34_0604')
    # epoch = '16-7.197602.pt'
    model_folder = os.path.join(model_root, 'ResUNet34_0616')
    epoch = '44-7.613293.pt'
    # TestClsSeg(model, device, model_folder, epoch, data_type='train')
    # TestClsSeg(model, device, model_folder, epoch, data_type='val')
    # TestClsSeg(model, device, model_folder, epoch, data_type='test')
    # ComputeAUC(os.path.join(model_folder, 'Result'), data_type='test')
    # ComputeClassAUC(os.path.join(model_folder, 'Result'), data_type='pz')
    # ComputeClassAUC(os.path.join(model_folder, 'Result'), data_type='cg')
    # ComputeClassAUC(os.path.join(model_folder, 'Result'), data_type='u')
    # ComputeClassAUC(os.path.join(model_folder, 'Result'), data_type='asmf')
    ComputeDice(os.path.join(model_folder, 'Result'))