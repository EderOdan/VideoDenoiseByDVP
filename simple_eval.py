import os
import argparse
import torch
import torch.nn as nn
from models.network import UNet as Net
import numpy as np
from utils import open_sequence, variable_to_cv2_image

import cv2


def test(**args):
    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    # Sets data type according to CPU or GPU modes

    device = torch.device('cuda')
    # Create models
    print('Loading models ...')
    model_temp = Net(in_channels=3, out_channels=3, init_features=32)
    state_temp_dict = torch.load(args['model_file'], map_location=device)
    device_ids = [0]
    model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    # Load saved weights
    model_temp.load_state_dict(state_temp_dict, strict=False)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    with torch.no_grad():
        # # process data
        # seq, _, _ = open_sequence(args['test_path'], False, expand_if_needed=False, max_num_fr=100)
        # seq = torch.from_numpy(seq).to(device)
        # frames,C,W,H = seq.shape
        # for i in frames:
        #     frame = seq[i,:,:,:].unsqueeze(0)
        #     de_frame = model_temp(frame)
        #
        for idx, files in enumerate(os.listdir(args['test_path'])):
            test_image = os.path.join(args['test_path'], files)
            test_save = os.path.join(args['save_path'], files)
            print(test_image)
            print(test_save)
            bgr_img = cv2.imread(test_image)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            rgb_img = np.expand_dims(rgb_img, 0)
            rgb_img = torch.from_numpy(rgb_img).permute(
                0, 3, 1, 2).float().div(255.0).clamp(0., 1.).to(device)
            print(rgb_img.shape)
            tensor_denoisy = model_temp(rgb_img)
            print(tensor_denoisy.shape)
            out_img = tensor_denoisy.cpu() * 255
            out_img = out_img.clamp(0, 255).type(torch.uint8)[
                0].permute(1, 2, 0).numpy()
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            print(out_img.shape)
            cv2.imwrite(test_save, out_img)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise")
    parser.add_argument("--model_file", type=str,
                        default="logstets/net.pth",
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--test_path", type=str, default="../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/noise_input",
                        help='path to sequence to denoise')
    parser.add_argument("--save_path", type=str, default='testData',
                        help='where to save outputs as png')

    argspar = parser.parse_args()
    print("\n### Testing MWCNN_sRGB model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    test(**vars(argspar))
