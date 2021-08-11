import torch
from torch import nn
import torchvision.models
from collections import OrderedDict

from pytorch_lightning import LightningModule

from .skipLSTM import SkipLSTM

class Resnet18_SkipLSTM(LightningModule):
    """
    Resnet18 with SkipLSTM module

    Hyperparameters: 
    hidden_size : SkipLSTM hidden_size
    n_layers : SkipLSTM n_layers
    num_classes: number of output classes
    freeze_cnn_before : Freezes all CNN pretrained layer before this, If None then doesn't freeze
                      Options: layer1, layer2, layer3, layer4
    """
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn = torchvision.models.resnet18(pretrained=True)

        # Remove the last fc layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze all layers before 'freeze_cnn_before'
        freeze_cnn_before = self.hparams.freeze_cnn_before
        if freeze_cnn_before is not None:
            for name, layer in self.cnn.named_children():
                if name != freeze_cnn_before:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    break

        self.lstm = SkipLSTM(input_size=512, hidden_size=self.hparams.hidden_size, n_layers=self.hparams.n_layers)     
        self.fc = nn.Sequential(nn.Linear(self.hparams.hidden_size, 256),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, self.hparams.num_classes))
        
    def forward(self, x):
        batch_size, c, frames, h, w = x.size()
        
        # CNN layer
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(batch_size * frames, c, h, w)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = x.contiguous().view(batch_size , frames , x.size(-1))
        
        # LSTM Layer
        x = self.lstm(x)[0]
        x = x[:, -1, :] # Get the output of the last cell

        # FC Layer
        x = self.fc(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    from torchsummary import summary

    video_model = Resnet18_SkipLSTM(num_classes=2, 
                                hidden_size=256,
                                n_layers=2,
                                freeze_cnn_before='layer3'
                                )
    print("Number of parameters ", video_model.num_params())