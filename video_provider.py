import torch
import glob
import os
import cv2
import numpy as np
import random
from utils import open_sequence,batch_psnr,variable_to_cv2_image
from torch.utils.data import Dataset, DataLoader
IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def get_imagenames(seq_dir, pattern=None):
    """ Get ordered list of filenames
    """
    files = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))

    # filter filenames
    if not pattern is None:
        ffiltered = []
        ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
        files = ffiltered
        del ffiltered
    # sort filenames alphabetically
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files
class Video_Provider_For_Davis(Dataset):
    def __init__(self, gt_path,process_path,sigma_seed= 0 ,sigma = 25):
    # def __init__(self, gt_path,process_path):
        if not os.path.exists(gt_path):
            raise("file_name is not valid")
        # self.aug_mode = aug_mode
        self.gt_path = gt_path
        self.sigma_seed = sigma_seed
        self.sigma = sigma
        self.process_path = process_path
        self.files = get_imagenames(gt_path)
        self.gt_seq, _, _ = open_sequence(self.gt_path, False, expand_if_needed=False, max_num_fr=100)
        self.process_seq, _, _ = open_sequence(self.process_path, False, expand_if_needed=False,max_num_fr=100)
        self.gt_seq = np.concatenate([np.expand_dims(self.gt_seq[-1, :, :, :], 0), self.gt_seq], axis=0)
        self.process_seq = np.concatenate([np.expand_dims(self.process_seq[-1, :, :, :], 0), self.process_seq], axis=0)

    def __len__(self):
        return len(self.files)+1

    def __getitem__(self,index):
        noise=self.gt_seq[index,:,:,:]
        g_noise = np.random.normal(loc=0,scale=self.sigma/255.,size=noise.shape)
        # print(noise.shape)
        noise = noise + g_noise
        process_gt = self.process_seq[index,:,:,:]
        return noise, process_gt
class Video_Provider_For_IOCV(Dataset):
    def __init__(self, noise_path,process_path):
    # def __init__(self, gt_path,process_path):
        if not os.path.exists(noise_path):
            raise("file_name is not valid")
        # self.aug_mode = aug_mod
        self.noise_path = noise_path
        self.process_path = process_path
        self.files = get_imagenames(noise_path)
        self.noise_seq, _, _ = open_sequence(self.noise_path, False, expand_if_needed=False, max_num_fr=100)
        self.process_seq, _, _ = open_sequence(self.process_path, False, expand_if_needed=False,max_num_fr=100)
        #
        self.noise_seq = np.concatenate([np.expand_dims(self.noise_seq[-1, :, :, :], 0), self.noise_seq], axis=0)
        self.process_seq = np.concatenate([np.expand_dims(self.process_seq[-1, :, :, :], 0), self.process_seq], axis=0)
        print('the frames have been added to be odd')
    def __len__(self):
        return len(self.files)+1
    def __getitem__(self,index):
        noise = self.noise_seq[index, :, :, :]
        process_gt = self.process_seq[index, :, :, :]
        return noise, process_gt
def Davis_dataset_test():
    gt_path = '../data/aerobatics/gt'
    process_path = '../data/aerobatics/25/CBM3D_process'
    dataset = Video_Provider_For_Davis(gt_path=gt_path,process_path= process_path )
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    for index, (noise, process_gt) in enumerate(train_loader):
        print(noise.shape)
        #print(batch_psnr(noise,gt,1.))
        # if index == 1:
        #     out_img = noise.cpu() [0]* 255
        #     out_img = out_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
        #     # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        #     # noisyimg = noise.clamp(0., 1.).cpu().numpy()*(255.)
        #     # noisyimg = noisyimg.clip(0, 255).astype(np.uint8)
        #     # noisyimg = cv2.cvtColor(noisyimg, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('./testgt.png', out_img)
def IOCV_dataset_test():
    noise_path = '../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/noise_input'

    process_path = '../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/dncnn_process'
    dataset = Video_Provider_For_IOCV(noise_path = noise_path, process_path=process_path)
    print(len(dataset))
    print(dataset[46][0].shape)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    for index, (noise, process_gt) in enumerate(train_loader):
        print(noise.shape)
    #     if index == 1:
    #         out_img = noise.cpu()[0] * 255
    #         out_img = out_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
    #         cv2.imwrite('./testnoise.png', out_img)
    #         procee_img = process_gt.cpu()[0] * 255
    #         procee_img = procee_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
    #         cv2.imwrite('./testprocess.png', procee_img)
    #         out2_img = gt.cpu()[0] * 255
    #         out2_img = out2_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
    #         cv2.imwrite('./testgt.png', out2_img)
class ValDataset_IOCV(Dataset):
    def __init__(self, gt_path):
        # def __init__(self, gt_path,process_path):
        if not os.path.exists(gt_path):
            raise ("file_name is not valid")
        # self.aug_mode = aug_mode
        self.gt_path = gt_path
        self.files = get_imagenames(gt_path)
        self.gt_seq, _, _ = open_sequence(self.gt_path, False, expand_if_needed=False, max_num_fr=100)
        self.gt_seq = np.concatenate([np.expand_dims(self.gt_seq[-1, :, :, :], 0), self.gt_seq], axis=0)
        self.frames = len(self.files)

    def __len__(self):
        return self.frames+1
    def __getitem__(self,index):
        gt = self.gt_seq[index,:,:,:]
        return gt

if __name__ == '__main__':
    Davis_dataset_test()

    # gt_path = '../data/IOCV/HUAWEI_HONOR_6X_FC_S_60_INDOOR_V1_1/gt'
    # dataset = ValDataset_IOCV(gt_path=gt_path)
    # print(len(dataset))
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # for index, gt in enumerate(train_loader):
    #     print(gt.shape)
                # if index == 1:
        #     out_img = noise.cpu()[0] * 255
        #     out_img = out_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
        #     cv2.imwrite('./testnoise.png', out_img)
        #     procee_img = process_gt.cpu()[0] * 255
        #     procee_img = procee_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
        #     cv2.imwrite('./testprocess.png', procee_img)
        #     out2_img = gt.cpu()[0] * 255
        #     out2_img = out2_img.clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
        #     cv2.imwrite('./testgt.png', out2_img)