from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import time
import cv2
import logging
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import models.network as unet
from image_warp import image_warp
from vgg import VGG19
from pwc import estimate as pwcnet

from utils import batch_psnr, init_logger_test, \
    variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger

VGG_19 = VGG19(requires_grad=False).to('cuda')


def batch_psnr(img, imclean, data_range):
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                             data_range=data_range)
    return psnr / img_cpu.shape[0]

# some functions
# define loss function
def compute_error(real, fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake - real))
def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std


def Lp_loss(x, y):
    vgg_real = VGG_19(normalize_batch(x))
    vgg_fake = VGG_19(normalize_batch(y))
    p0 = compute_error(normalize_batch(x), normalize_batch(y))

    content_loss_list = []
    content_loss_list.append(p0)
    feat_layers = {'conv1_2': 1.0 / 2.6, 'conv2_2': 1.0 / 4.8, 'conv3_2': 1.0 / 3.7, 'conv4_2': 1.0 / 5.6,
                   'conv5_2': 10.0 / 1.5}

    for layer, w in feat_layers.items():
        pi = compute_error(vgg_real[layer], vgg_fake[layer])
        content_loss_list.append(w * pi)

    content_loss = torch.sum(torch.stack(content_loss_list))

    return content_loss


loss_L2 = torch.nn.MSELoss()
loss_L1 = torch.nn.L1Loss()


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # nn.init.kaiming_normal_(module.weight)
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()





def train_dvp(**args):
    # Start time
    start_time = time.time()

    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net = unet.UNet(in_channels=3, out_channels=3, init_features=32)
    net = nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000,8000], gamma=0.5)


    noise_seq, _, _ = open_sequence(args['input_path'], False, expand_if_needed=False, max_num_fr=100)
    noise_seq = np.concatenate([np.expand_dims(noise_seq[-1, :, :, :], 0), noise_seq], axis=0)
    process_seq, _, _ = open_sequence(args['processed_path'], False, expand_if_needed=False, max_num_fr=100)
    process_seq = np.concatenate([np.expand_dims(process_seq[-1, :, :, :], 0), process_seq], axis=0)
    gt_seq, _, _ = open_sequence(args['gt_path'], False, expand_if_needed=False, max_num_fr=100)
    gt_seq = np.concatenate([np.expand_dims(gt_seq[-1, :, :, :], 0), gt_seq], axis=0)


    data_seq = np.concatenate([noise_seq,process_seq,gt_seq],axis=1)

    data_seq = torch.from_numpy(data_seq).float().to(device)
    seq_time = time.time()
    numframes, C, H, W = data_seq.size()


    psnr_epochs = []
    # random
    initialize_weights(net)
    step = 0
    for epoch in range(0, args['max_epoch']):
        frame_id = 0
        # if os.path.isdir("{}/{:04d}".format(args['save_path'], epoch)):
        #     continue
        # else:
        #     os.makedirs("{}/{:04d}".format(args['save_path'], epoch))
        for fridx in range(numframes - 1):
            # net_in.shape=[2,3,480,832]
            # net_in, net_gt = data_seq[fridx:fridx + 2, :3, :, :], data_seq[fridx:fridx + 2, 3:6, :, :]
            # # net_in, net_gt = noise_seq[fridx:fridx + 1, :, :, :], process_seq[fridx:fridx + 1, :, :, :]
            # # net_in_next, net_gt_next, = noise_seq[fridx + 1, :, :, :].unsqueeze(0), gt_seq[fridx + 1, :, :,
            # #                                                                         :].unsqueeze(0)
            # deframe_cur = net(net_in)
            # # deframe_nex = net(net_in_next)
            # crt1_loss = loss_L2(deframe_cur, net_gt)
            # crt1_loss.requres_grad = True
            # # crt2_loss = loss_L2(deframe_nex, net_gt_next)
            # loss = crt1_loss
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # # print('finish {} step'.format(step))
            # frame_id += 1
            # step += 1
            # net_in.shape=[2,3,480,832]
            net_in, net_gt = data_seq[fridx:fridx + 2, :3, :, :], data_seq[fridx:fridx + 2, 3:6, :, :]
            deframe = net(net_in)

            tenFirst = deframe[0, :, :, :]
            tenSecond = deframe[1, :, :, :]
            tenOutput = pwcnet(tenFirst,tenSecond).detach().permute(0,2,3,1).numpy()
            tenSecond = tenSecond.cpu().unsqueeze(0).permute(0,2,3,1).detach().numpy()*(255.0)
            deformed_nearest = image_warp(tenSecond.copy(), tenOutput, mode='bilinear')


            pwc_loss = Lp_loss(tenFirst.unsqueeze(0)/255.,torch.from_numpy(deformed_nearest).float().to(device).permute(0,3,1,2)/255.)
            crt_loss = Lp_loss(deframe, net_gt) +pwc_loss
            loss = crt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('finish {} step'.format(step))
            frame_id += 1
            step += 1
        psnr = 0
        for fridx in range(numframes-1):
            net_in= data_seq[fridx, :3, :, :].unsqueeze(0)
            with torch.no_grad():
                prediction = net(net_in)
            img_cpu = prediction.squeeze(0).data.cpu().numpy().astype(np.float32)
            imgclean = data_seq[fridx,6:9,:,:].data.cpu().numpy().astype(np.float32)
            psnr += compare_psnr(imgclean, img_cpu, data_range=1.)
        cur_epoch_psnr = psnr/ numframes
        print('finish {} epoch, {},iters ,current loss:{},current psnr:{}'.format(epoch, step, loss, cur_epoch_psnr))
        logger.info('finish {} epoch, {},iters ,current loss:{},current psnr:{}'.format(epoch, step, loss, cur_epoch_psnr))

        # if epoch % args['save_freq'] == 0:
        #     if not os.path.isdir("{}/{:04d}".format(args['save_path'], epoch)):
        #         os.makedirs("{}/{:04d}".format(args['save_path'], epoch))
        #     for fridx in range(numframes):
        #         net_in, net_gt = noise_seq[fridx, :, :, :].unsqueeze(0), gt_seq[fridx, :, :, :].unsqueeze(0)
        #         print("Test: {}-{} \r".format(fridx, numframes))
        #         with torch.no_grad():
        #             prediction = net(net_in)
        #         net_in = net_in.permute(0, 2, 3, 1).cpu().numpy()
        #         net_gt = net_gt.permute(0, 2, 3, 1).cpu().numpy()
        #         prediction = prediction.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        #         cv2.imwrite("{}/{:04d}/{:05d}.jpg".format(args['save_path'], epoch, fridx),
        #                     np.uint8(prediction.clip(0, 1) * 255.0))
    print(psnr_epochs)
    # psnr_epochs_sorted = psnr_epochs.sort(key = lambda d:d[1],reverse=True)
    # print(psnr_epochs_sorted)




def test_dvp():
    pass


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='DVP+', type=str, help="Name of model")
    parser.add_argument("--save_freq", default=2, type=int, help="save frequency of epochs")
    parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
    parser.add_argument("--max_epoch", default=200, type=int, help="The max number of epochs for training")
    # parser.add_argument("--input_path", default='./test/noise_input', type=str, help="dir of the noise video")
    # parser.add_argument("--processed_path", default='./test/dncnn_process', type=str,
    #                     help="dir of processed gt video")
    parser.add_argument("--input_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/noise_input', type=str,
                        help="dir of the noise video")
    parser.add_argument("--processed_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/dncnn_process', type=str,
                        help="dir of processed gt video")
    parser.add_argument("--gt_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/gt', type=str, help="dir of processed gt video")
    parser.add_argument("--save_path", default='./result', type=str, help="dir of output video")
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument('--eval', '-e', action='store_true', help='whether to work on the eval mode')

    argspar = parser.parse_args()
    # use CUDA?
    argspar.cuda = argspar.use_gpu and torch.cuda.is_available()

    print("\n### modify dvp for video denoising by odan  ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    # if not argspar.eval:
    train_dvp(**vars(argspar))
    # else:
    #     with torch.no_grad():
    #         test_dvp(**vars(argspar))
