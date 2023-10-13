import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from argparse import ArgumentParser
from tqdm import tqdm
import os

from utils.data import MyDataset
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA
from model.model_HoLoCoNet import HoLoCoNet


device = torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    # Setting parameters
    parser = ArgumentParser(description='Implement of RDIAN model')
    # Size of images
    parser.add_argument('--path', type=str, default='./datasets/', help='path of the dataset')
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST', help='dataset name: NUDT-SIRST or IRSTD-1k')
    parser.add_argument('--checkpoint', type=str, default='./result/holoconet_nudt.pth', help='path of the checkpoint')
    parser.add_argument('--mode', type=str, default='val', help='val or train')

    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')

    args = parser.parse_args()
    return args


class test(object):
    def __init__(self, args):
        self.args = args

        self.unloader = transforms.ToPILImage()

        ## dataset
        testset = MyDataset(args)
        self.test_data_loader = Data.DataLoader(testset, batch_size=1, num_workers=8, shuffle=False)

        ## model
        self.net = HoLoCoNet()
        weights = torch.load(args.checkpoint)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        self.net.load_state_dict(weights_dict, strict=True)
        self.net.eval()
        self.net = self.net.cuda()

        ## evaluation metrics
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(1, 20)
        self.best_iou = 0
        self.best_nIoU = 0

    def testing(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()

        tbar = tqdm(self.test_data_loader)

        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output1 = output.cpu()

            self.iou_metric.update(output1, labels)
            self.nIoU_metric.update(output1, labels)
            self.PD_FA.update(output, labels, 512, 512)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()

            tbar.set_description('IoU:%f, nIoU:%f' % (IoU, nIoU))
        FA, PD = self.PD_FA.get(len(self.test_data_loader), 512, 512)
        print('FA: ')
        print(FA[0])
        print('\n')
        print('PD: ')
        print(PD[0])


if __name__ == '__main__':
    args = parse_args()

    test = test(args)
    test.testing()
    print("over")
