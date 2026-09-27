from data.dataloaders import create_seg_dataloaders
from env_setup import LocalVariables

import torch

lv = LocalVariables()

train_dataloader, valid_dataloader, test_dataloader = create_seg_dataloaders(lv.source_seg, lv.target_seg, batch_size=4)

# Test to make sure the dataloader is working
test_iteration = next(iter(train_dataloader))
source_access = test_iteration['image']
target_access = test_iteration['mask']
pass