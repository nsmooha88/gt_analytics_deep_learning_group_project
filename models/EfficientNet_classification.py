import torch
import torch.nn as nn
import torchvision


class EfficientNet(nn.Module):
    def __init__(self, num_class=15, dropout=0.2):
        super(EfficientNet, self).__init__()
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        self.model_name = "efficientnet"
        self.dropout = dropout
        self.num_class = num_class
        
        # Needed to avoid hash error
        def get_state_dict(self, *args, **kwargs):
            kwargs.pop("check_hash")
            return torch.hub.load_state_dict_from_url(self.url, *args, **kwargs)
        torchvision.models._api.WeightsEnum.get_state_dict = get_state_dict
        pretrained_efficientnet = torchvision.models.efficientnet_b4(weights = "DEFAULT")
        for param in pretrained_efficientnet.features.parameters():
            param.requires_grad = True
        
        self.pretrained_layer = nn.Sequential(
            *nn.ModuleList(pretrained_efficientnet.children())[0])
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1792,kernel_size=(3,3), stride=(1,1), padding=(1,1), out_channels=256, bias=True, device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), out_channels=128, bias=True, device=self.device),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), out_channels=64, bias=True, device=self.device),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p = self.dropout)
        )

        self.final_classifier = nn.Sequential(
            nn.LazyLinear(out_features=512, bias=True, device=self.device),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=512, out_features=256, bias=True, device=self.device),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=256, out_features=128, bias=True, device=self.device),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features=128, out_features=self.num_class, bias=True, device=self.device)
        )

    def forward(self, x):
        outs = self.pretrained_layer(x)
        outs = self.cnn(outs)
        outs = torch.flatten(outs,start_dim=1)
        outs = self.final_classifier(outs)
        return outs
