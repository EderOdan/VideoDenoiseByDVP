from vgg import VGG19
from pwc import estimate as pwcnet
import torch.nn as nn
import torch

VGG_19 = VGG19(requires_grad=False).to('cuda')
def compute_error(real, fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake - real))
# def normalize_batch(batch):
#     # Normalize batch using ImageNet mean and std
#     mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
#     std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
#     return (batch - mean) / std
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

L2Loss = nn.MSELoss(reduction='sum')

