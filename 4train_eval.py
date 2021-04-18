import os
import argparse
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from video_provider import Video_Provider_For_Davis,Video_Provider_For_IOCV,ValDataset_IOCV
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from models.network import UNet as unet
import torch.nn as nn
from train_common import resume_training, lr_scheduler, log_train_psnr, save_model_checkpoint
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from vgg import VGG19
from utils import batch_psnr, init_logger_test, \
    variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger
# def validate_and_log(noiseframe,dataset_val ,writer, epoch, lr=0.0001, logger):
#     t1 = time.time()
#     psnr_val = 0
#     with torch.no_grad():
#         for seq_val in dataset_val:
# define loss function
VGG_19 = VGG19(requires_grad=False).to('cuda')
def compute_error(real, fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake - real))
def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std


def Lp_loss(x, y):
    vgg_real = VGG_19(x)
    vgg_fake = VGG_19(y)
    p0 = compute_error(x, y)

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
def main(**args):
    # Load dataset
    print('> Loading datasets ...')
    dataset = Video_Provider_For_IOCV(noise_path=args['input_path'], process_path=args['processed_path'])

    loader_train = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    dataset_val = ValDataset_IOCV(gt_path=args['gt_path'])
    # fix ~
    num_minibatches = len(dataset)
    args['save_every'] = len(dataset)

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPU devices
    device_ids = [0]
    # torch.backends.cudnn.benchmark = True  # CUDNN optimization

    # Create model
    model = unet(in_channels=3, out_channels=3, init_features=32)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    initialize_weights(model)

    # Define loss
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.MSELoss()
    criterion.cuda()

    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # fastdvdnet learning rate setting
        # # Set learning rate
        # current_lr, reset_orthog = lr_scheduler(epoch, args)
        # if reset_orthog:
        #     training_params['no_orthog'] = True
        #
        # # set learning rate in optimizer
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = current_lr
        # print('\nlearning rate %f' % current_lr)


        # train
        for i, (net_in, process_gt) in enumerate(loader_train):

            # Pre-training step
            model.train()

            # When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
            optimizer.zero_grad()

            # convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
            N, _, H, W = net_in.size()

            # Send tensors to GPU
            net_in = net_in.cuda(non_blocking=True)
            process_gt = process_gt.cuda(non_blocking=True)
            # gt = gt.cuda(non_blocking=True)

            # Evaluate model and optimize it
            out_train = model(net_in)

            # Compute loss
            loss = Lp_loss(process_gt, out_train)
            # loss = criterion(process_gt, out_train)
            loss.backward()
            optimizer.step()

            # # Results
            # if training_params['step'] % args['save_every'] == 0:
            #     # Apply regularization by orthogonalizing filters
            #     if not training_params['no_orthog']:
            #         model.apply(svd_orthogonalization)
            #
            #     # Compute training PSNR
            #     log_train_psnr(out_train, process_gt, loss, writer, epoch, i,num_minibatches , training_params)

            # update step counter
            training_params['step'] += 1



        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()
        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

        # Validation and log images
        # validate_and_log(noiseframe = out_train,dataset_val = dataset_val,writer=writer, epoch=epoch, lr=0.0001, logger=logger)
        t1 = time.time()
        psnr_val = 0
        with torch.no_grad():
            for index, (net_in, process_gt) in enumerate(dataset):
                denoise_frame = model(torch.from_numpy(net_in).unsqueeze(0).cuda(non_blocking=True))
                denoise_frame = denoise_frame.cpu().numpy().astype(np.float32)[0]
                val_frame = dataset_val[index]
                psnr_val += compare_psnr(val_frame, denoise_frame, data_range=1.)
        cur_epoch_psnr = psnr_val/len(dataset)
        print("[epoch %d] PSNR_val: %.4f,loss: %.4f" % (epoch + 1, cur_epoch_psnr,loss))
        logger.info("[epoch %d] PSNR_val: %.4f,loss: %.4f" % (epoch + 1, cur_epoch_psnr, loss))











    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger file
    close_logger(logger)




if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='DVP+', type=str, help="Name of model")
    parser.add_argument("--save_freq", default=2, type=int, help="save frequency of epochs")

    # save path
    # parser.add_argument("--input_path", default='../data/aerobatics/25/noise_input', type=str,help="dir of the noise video")
    # parser.add_argument("--processed_path", default='../data/aerobatics/25/dncnn_process', type=str, help="dir of processed gt video")
    # parser.add_argument("--gt_path", default='../data/aerobatics/gt', type=str, help="dir of processed gt video")
    parser.add_argument("--input_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/noise_input',
                        type=str, help="dir of the noise video")
    parser.add_argument("--processed_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/CBM3D_process',
                        type=str, help="dir of processed gt video")
    parser.add_argument("--gt_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/gt', type=str,
                        help="dir of gt video")
    parser.add_argument("--save_path", default='./result', type=str, help="dir of output video")
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")

    # what?
    parser.add_argument("--no_orthog", action='store_true', help="Don't perform orthogonalization as regularization")


    # sure(fastdvdnet)
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Number of training epochs to save state")
    parser.add_argument("--save_every", type=int, default=None, help="Number of training steps to log psnr and perform orthogonalization")
    parser.add_argument("--milestone", nargs=2, type=int, default=[30, 80], help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
    parser.add_argument("--epochs", default=100, type=int, help="The max number of epochs for training")
    parser.add_argument("--log_dir", type=str, default="./logstets", \
                        help='path of log files')
    parser.add_argument("--lr", type=float, default=1e-3, \
                        help="Initial learning rate")
    parser.add_argument("--resume_training", "--r", action='store_true', \
                        help="resume training from a previous checkpoint")

    argspar = parser.parse_args()
    # use CUDA?
    argspar.cuda = argspar.use_gpu and torch.cuda.is_available()

    print("\n### modify dvp for video denoising by odan  ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    main(**vars(argspar))



