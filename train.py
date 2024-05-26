import os
import argparse
import torch
import torch.nn as nn
from models.model import MODEL

from torch.autograd import Variable
from utils.accuracy import *
import random
import cv2
import torch.nn.functional as F
from utils.evaluation import cal_kl, cal_sim, cal_nss
import time

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data_path') #data_path(In addition, there is GT_path that needs to be set on line 214 of this file.)
parser.add_argument('--save_root', type=str, default='save_models')
parser.add_argument("--divide", type=str, default="Unseen") #"Seen" or "Unseen" or "HICO-IIF"
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
#  train
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=13)
parser.add_argument('--pretrain', type=str, default='True')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--threshold', type=float, default='0.2')
parser.add_argument("--D", type=int, default=512)
parser.add_argument('--show_step', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')

# test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args(args=[])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr
if args.divide == "Seen":  # Seen
    args.num_classes = 36

elif args.divide == "Unseen":
    args.num_classes = 25  # Unseen
    
else:
    args.num_classes = 10 # HICO-IIF

args.exocentric_root = os.path.join(args.data_root, args.divide, "trainset", "exocentric")
args.egocentric_root = os.path.join(args.data_root, args.divide, "trainset", "egocentric")
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
time_str = time.strftime('%Y%m%d_%H%I%M', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)  # save model parameters

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

str_1 = ""
for key, value in dict_args.items():  # Write all experiment related parameters
    str_1 += key + "=" + str(value) + "\n"
with open(os.path.join(args.save_path, "write.txt"), "a") as f:
    f.write(str_1)

f.close()


def normalize_map(atten_map):  # normalized image
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)

    atten_norm = cv2.resize(atten_norm, dsize=(args.crop_size, args.crop_size))
    return atten_norm

def get_optimizer(model, args):
    lr = args.lr
    weight_list = []  # wight
    bias_list = []  # bias
    last_weight_list = []  # the last weight
    last_bias_list = []  # the last bias
    l_list =[]
    no_list = []
    for name, value in model.named_parameters():
        if name not in par:
            if 'fc' in name:  # The weight value stored in the last fully connected layer
                if 'weight' in name:
                    last_weight_list.append(value)
                elif 'bias' in name:
                    last_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
                else:
                    l_list.append(value)
        else: # The weight value stored in the text encoder
            no_list.append(value)
    # Different layers have different learning rates
    optmizer = torch.optim.SGD([{'params': weight_list,
                                 'lr': lr},
                                {'params': bias_list,
                                 'lr': lr * 2},
                                {'params': l_list,
                                 'lr': lr*2},
                                {'params': last_weight_list,
                                 'lr': lr * 10},
                                {'params': last_bias_list,
                                 'lr': lr * 20},
                               {'params': no_list,
                                 'lr': lr *2}],
                               momentum=args.momentum,  # 0.9
                               weight_decay=args.weight_decay,  # 5e-4
                               nesterov=True)
    return optmizer


if __name__ == '__main__':
    if args.phase == 'train':

        import datatrain
        import datatest

        trainset = datatrain.TrainData(exocentric_root=args.exocentric_root,
                             egocentric_root=args.egocentric_root,
                             resize_size=args.resize_size, crop_size=args.crop_size, divide=args.divide)

        MyDataLoader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)

        testset = datatest.TrainData(egocentric_root=args.test_root, crop_size=args.crop_size, divide=args.divide, mask_root=args.mask_root)
        MytestDataLoader = torch.utils.data.DataLoader(dataset=testset,
                                                       batch_size=args.test_batch_size,
                                                       shuffle=False,
                                                       num_workers=args.test_num_workers,
                                                       pin_memory=True)
        model, par = MODEL(args, num_classes=args.num_classes,
                           pretrained=True, n=3, D=args.D)
        model= model.cuda()
        model.train()
        optimizer = get_optimizer(model, args)
        loss_func = nn.CrossEntropyLoss().cuda()
        epoch_loss = 0
        correct = 0
        total = 0
        acc_sum = 0
        a = 0
        best_kld = 1000
        print('Train')
        #args.epochs
        for epoch in range(0, args.epochs):
            model.train()
            ##  accuracy
            exo_acc = AverageMeter()
            ego_acc = AverageMeter()
            epoch_loss = 0
            aveGrad = 0
            for step, (exocentric_image, egocentric_image, label, name) in enumerate(MyDataLoader):
                label = label.cuda(non_blocking=True)
                egocentric_image = egocentric_image.cuda(
                    non_blocking=True)
                ##  backward
                optimizer.zero_grad()
                exocentric_images = Variable(torch.stack(exocentric_image, dim=1)).cuda(non_blocking=True)
                exo_out, ego_out, text, image= model(exocentric_images, egocentric_image, label, name)
                label = label.long()
                loss_cls, loss_dist, loss_att = model.get_loss(label)
                
                loss_i = loss_func(image, label)
                # loss L_clip
                clip_loss = loss_i
                
                # the final loss
                loss = loss_cls.cuda()+ loss_dist.cuda()+ clip_loss.cuda()+loss_att.cuda()

                loss.backward()
                optimizer.step()
                
                #  count_accuracy
                cur_batch = label.size(0)
                epoch_loss += loss.item()
                b, h= exo_out.shape
                exo_out = exo_out.view(b//3, -1, h)
                exo_cls_acc = 100. * compute_cls_acc(exo_out[:, 0], label)
                ego_cls_acc = 100. * compute_cls_acc(ego_out, label)
                exo_acc.updata(exo_cls_acc, cur_batch)
                ego_acc.updata(ego_cls_acc, cur_batch)

                if (step + 1) % args.show_step == 0 or step==0:
                    print(
                        '{} \t Epoch:[{}/{}]\tstep:[{}/{}] \t cls_loss: {:.3f}\t att_loss: {:.3f}\tclip_loss: {:.3f}\tdist_loss: {:.3f}\t exo_acc: {:.2f}% \t ego_acc: {:.2f}%'.format(
                            args.phase, epoch + 1, args.epochs, step + 1, len(MyDataLoader), loss_cls.item(), loss_att.item(), clip_loss.item(), loss_dist.item(),
                            exo_acc.avg, ego_acc.avg
                        ), flush=True)
            KLs = []
            model.eval()
            masks = torch.load("Unseen_gt.t7") # GT_path
            for step, (egocentric_image, label, mask_path, name) in enumerate(MytestDataLoader):
                label = label.cuda(non_blocking=True)
                egocentric_image = egocentric_image.cuda(
                    non_blocking=True)
                cam, cam1 = model.get(egocentric_image, label, name)
                cam = cam[0].cpu().detach().numpy()
                cam1 = cam1[0].cpu().detach().numpy()

                names = mask_path[0].split("/")
                key = names[-3] + "_" + names[-2] + "_" + names[-1]
                mask = masks[key]
                mask = mask / 255.0
                
                # cam refined
                mask = cv2.resize(mask, (args.crop_size, args.crop_size))
                cam = normalize_map(cam)
                cam1 = normalize_map(cam1)
                cam1[cam <args.threshold]=0
                final_cam = normalize_map(cam+cam1)
                kld = cal_kl(final_cam, mask)

                KLs.append(kld)
            mKLD = sum(KLs) / len(KLs)
            print("epoch=" + str(epoch) + " " + "mKLD = " + str(mKLD), flush=True)
            if mKLD < best_kld:
                best_kld = mKLD

                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth.tar'))
            