import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):
    def __init__(self, num_class=15, dropout=0.2):
        super(VGG, self).__init__()
        self.model_name = "vgg"
        self.num_class = num_class
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pretrained_vgg = torchvision.models.vgg19(weights = "DEFAULT")
        for param in self.pretrained_vgg.features.parameters():
            param.requires_grad = True

        self.pretrained_layer = nn.Sequential(
            *nn.ModuleList(self.pretrained_vgg.children())[0])
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = 512,kernel_size=(3,3), stride=(1,1), padding=(1,1),out_channels=256,device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, kernel_size=(3,3), stride=(1,1), padding=(1,1),out_channels=128,device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128,kernel_size=(3,3), stride=(1,1), padding=(1,1),out_channels=64,device=self.device),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64,device=self.device),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Dropout(p = self.dropout)
        )
        
        self.final_classifier = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=128, out_features=self.num_class)
        )

    def forward(self, x):
        outs = self.pretrained_layer(x)
        outs = self.cnn(outs)
        outs = torch.flatten(outs,start_dim=1)
        outs = self.final_classifier(outs)
        return outs
