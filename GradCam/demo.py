from __future__ import print_function

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from CnnTools.T4T.Utility.Data import *
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.Visualization import FusionImage

from GradCam.grad_cam import GradCAM
from GradCam.grad_cam_main import save_gradcam

from SegModel.ResNet50 import ModelRun


def demo_my(model, input_list, input_class):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model.eval()
    target_layers = ["layer4"]
    target_class = torch.argmax(input_class)

    gcam = GradCAM(model=model)
    logits, probs = gcam.forward(input_list)

    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(192, 192))

        print("\t#{} ({:.5f})".format(target_class, float(probs[:, 1])))

        gradcam = save_gradcam(
            gcam=regions[0, 0, ...],
        )

    return probs[:, 1], gradcam


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='PZ'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='CG'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='U'), is_input=False)
    data.AddOne(Label(data_root + '/class_label.csv', label_tag='AMSF'), is_input=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches



if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/ResNet34_0604/16-7.197602.pt'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    output_dir = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'

    model = ModelRun(1, 4, res_num=34).to(device)
    model.load_state_dict(torch.load(model_root))
    model.to(device)

    test_df = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format('test')))
    test_list = test_df.values.tolist()[0]
    data_loader, data_batches = _GetLoader(test_list, None, (200, 200), 1, False)

    for ind, (inputs, outputs) in enumerate(data_loader):

        if outputs[2] == 0:
            input_class = torch.tensor([1, 0]).long()
        else:
            input_class = torch.tensor([0, 1]).long()

        prob, gradcam = demo_my(model, inputs.to(device), input_class.to(device))

        merged_image = FusionImage(Normalize01(np.squeeze(inputs)),
                                   Normalize01(np.squeeze(gradcam)), is_show=False)

        plt.suptitle("label: {}, pred: {:.3f}".format((outputs[0]), float((prob).cpu().data)))
        plt.subplot(121)
        plt.axis('off')
        plt.imshow(np.squeeze(inputs), cmap='gray')
        plt.subplot(122)
        plt.axis('off')
        plt.imshow(np.squeeze(merged_image), cmap='jet')

        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.00, hspace=0.01)

        # plt.savefig(r'/home/zhangyihong/Documents/ProstateECE/Paper/' + str(i) + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.00)
        # plt.savefig(os.path.join(output_dir, '{}_without.jpg'.format(i)), format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.00)
        plt.show()
        plt.close()

