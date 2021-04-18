import argparse
import torch
from utils import batch_psnr,open_sequence
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/noise_input',
                        type=str, help="dir of the noise video")
    parser.add_argument("--processed_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/dncnn_process',
                        type=str, help="dir of processed gt video")
    parser.add_argument("--gt_path", default='../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/gt', type=str,
                        help="dir of gt video")
    argspar = parser.parse_args()

    noise_seq, _, _ = open_sequence(argspar.processed_path, False, expand_if_needed=False, max_num_fr=100)
    gt_seq, _, _ = open_sequence(argspar.gt_path, False, expand_if_needed=False, max_num_fr=100)
    psnr = batch_psnr(torch.from_numpy(noise_seq).to('cuda'), torch.from_numpy(gt_seq).to('cuda'), 1.)
    print(psnr)