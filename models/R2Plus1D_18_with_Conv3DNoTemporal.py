import torch
from torch import nn
from torchvision.models import video as video_models
from collections import OrderedDict

from pytorch_lightning import LightningModule

from .blocks import *

def make_layer(inplanes, block, conv_builder, planes, blocks, stride=1):
    downsample = None

    if stride != 1 or inplanes != planes * block.expansion:
        ds_stride = conv_builder.get_downsample_stride(stride)
        downsample = nn.Sequential(
            SeperableConv3D(inplanes, planes * block.expansion,
                      kernel_size=1, stride=ds_stride, bias=False),
            nn.BatchNorm3d(planes * block.expansion)
        )
    layers = []
    layers.append(block(inplanes, planes, conv_builder, stride, downsample))

    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, conv_builder))

    return nn.Sequential(*layers)

class R2Plus1D_18_with_Conv3DNoTemporal(LightningModule):
    """
    Pretrained r2plus1d-18 with Conv3DNoTemporal

    Hyperparameters: 
    num_classes - number of output classes
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.pretrained_block = video_models.r2plus1d_18(pretrained=True)
        self.pretrained_block = nn.Sequential(OrderedDict([
                            ("stem", self.pretrained_block.stem),
                            ("layer1", self.pretrained_block.layer1)
                            ])
                        )
                                        
        self.conv_block = nn.Sequential(
            nn.Conv3d(64, 64, (3,3,3), stride=(2,2,2), bias = False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, (3,3,3), stride=(2,2,2), bias = False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        self.layer1 = make_layer(128, BasicBlock, Conv3DNoTemporal, 256, 2, stride=2) 
        self.layer2 = make_layer(256, BasicBlock, Conv3DNoTemporal, 512, 1, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) 
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, self.hparams.num_classes)
    
    def forward(self, x):
        x = self.pretrained_block(x)
        x = self.conv_block(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.avgpool(x)
        x = self.fc(self.flatten(x))
        return x

if __name__ == '__main__':
    from torchsummary import summary

    video_model = R2Plus1D_18_with_Conv3DNoTemporal(num_classes=2)
    summary(video_model.cpu(), (3, 32, 128, 128), device='cpu')