import torch
import torch.nn as nn
import torch.nn.functional as F
from models.builder import EncoderDecoder as segmodel
from local_configs.NYUDepthv2.DFormer_Tiny import C
import torchvision
from thop import profile

print('==> Building model..')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = C
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
BatchNorm2d = nn.BatchNorm2d
model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, single_GPU=1).to(device)

# Model
dummy_input = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))




