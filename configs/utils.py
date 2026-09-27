import os, tempfile
import time
import glob
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from models.VGG_classification import VGG
from models.EfficientNet_classification import EfficientNet

from data.dataloaders import create_class_dataloaders

from training.focal_loss import FocalLoss, reweight

from ray import tune
from ray.air import session
from ray.train import Checkpoint, report
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def delete_files(folder, file=None):
    curr_dir = os.getcwd()

    if (folder=='/checkpoints/') | (folder=='/checkpoints/best/') :
        upd_dir = curr_dir + folder
        dir_list = os.listdir(upd_dir)
        if file is None:
            file = ".pth"
        else:
            file = file + ".pth"

    elif (folder=='/logging/') | (folder=='/logging/best/'):
        upd_dir = curr_dir + folder
        dir_list = os.listdir(upd_dir)
        if file is None:
            file = ".csv"
        else:
            file = file + ".csv"

    elif (folder=='/plots/') | (folder=='/plots/best/'):
        upd_dir = curr_dir + folder
        dir_list = os.listdir(upd_dir)
        if file is None:
            file = ".png"
        else:
            file = file + ".png"

    for item in dir_list:
        if item.endswith(file):
            os.remove(os.path.join(upd_dir, item))

def create_temp_df(params, idx):
    steps = params['steps']
    betas = params['betas']
    
    params['steps'] = str(params['steps'])
    params['betas'] = str(params['betas'])
    df_temp = pd.DataFrame(params, index=[idx])
    
    params['steps'] = steps
    params['betas'] = betas

    return df_temp

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_metrics(output, target):
    """Computes the accuracy, precision, recall, and auroc for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    target_matrix = torch.zeros(output.shape)
    target_matrix[torch.arange(output.shape[0]),target.cpu()] = 1
    output_matrix = torch.zeros(output.shape)
    output_matrix[torch.arange(output.shape[0]),output.argmax(dim=1).cpu()] = 1

    precision = torch.tensor(precision_score(target_matrix.cpu(), output_matrix.detach().cpu(), average = 'weighted', zero_division = 0), device = acc.device) 
    recall = torch.tensor(recall_score(target_matrix.cpu(), output_matrix.detach().cpu(), average = 'weighted', zero_division = 0), device = acc.device)
    f1 = torch.tensor(f1_score(target_matrix.cpu(), output_matrix.detach().cpu(), average = 'weighted', zero_division = 0), device = acc.device)
    auroc = torch.tensor(roc_auc_score(target_matrix.cpu(), output.detach().cpu(), average = 'samples',multi_class='ovr'), device = acc.device)

    return acc, precision, recall, f1, auroc

def adjust_learning_rate(optimizer, epoch, learning_rate, warmup, steps):
    epoch += 1
    if epoch <= warmup:
        lr = learning_rate * epoch / warmup
    elif epoch > steps[1]:
        lr = learning_rate * 0.01
    elif epoch > steps[0]:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch, data_loader, model, optimizer, criterion, stop_early = False):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    iter_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    auroc = AverageMeter()

    # Set model to training mode
    model.train()

    # Training Loop
    print('\nTraining:')
    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        data = data.to(device)
        target = target.to(device)
        model = model.to(device)

        out = model.forward(data)
        target_idx = target.argmax(dim=1)
        target_idx = target_idx.to(device)
        
        loss = criterion(out, target_idx)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc, batch_precision, batch_recall, batch_f1, batch_auroc = accuracy_metrics(out, target.max(dim=-1)[1])

        losses.update(loss.item(), out.shape[0])
        accuracy.update(batch_acc, out.shape[0])
        precision.update(batch_precision)
        recall.update(batch_recall)
        f1.update(batch_f1)
        auroc.update(batch_auroc)

        iter_time.update(time.time() - start)
        if idx % 5 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time: {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                   'Accur.: {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                   'Prec.: {precision.val:.4f} ({precision.avg:.4f})\t'
                   'Recall: {recall.val:.4f} ({recall.avg:.4f})\t'
                   'F1: {f1.val:.4f} ({f1.avg:.4f})\t'
                   'AUROC: {auroc.val:.4f} ({auroc.avg:.4f})').format(
                       epoch,
                       idx,
                       len(data_loader),
                       iter_time=iter_time,
                       losses=losses,
                       accuracy=accuracy,
                       precision=precision,
                       recall=recall,
                       f1=f1,
                       auroc=auroc))

    if not stop_early:    
        if (epoch+1)%5 == 0:
            delete_files(folder="/checkpoints/", file="epoch")
            torch.save(model.state_dict(), './checkpoints/' + model.model_name + '_' + str(epoch+1) + '_epoch.pth')
    else:
        torch.save(model.state_dict(), './checkpoints/' + model.model_name + '_' + str(epoch+1) + '_epoch.pth')

    results = {
        'loss':losses.avg,
        'accuracy':accuracy.avg,
        'precision':precision.avg,
        'recall':recall.avg,
        'f1':f1.avg,
        'auroc':auroc.avg
    }

    return results

def validate(epoch, valid_loader, num_class, model, criterion):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    iter_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    auroc = AverageMeter()
    cm = torch.zeros(num_class, num_class)

    # Set model to evaluation mode
    model.eval()
    
    # evaluation loop
    print('\nValidation:')
    for idx, (data, target) in enumerate(valid_loader):
        start = time.time()

        data = data.to(device)
        target = target.to(device)
        target_idx = target.argmax(dim=1)
        target_idx = target_idx.to(device)
        
        with torch.no_grad():
            out = model.forward(data)
            loss = criterion(out, target_idx)

        batch_acc, batch_precision, batch_recall, batch_f1, batch_auroc = accuracy_metrics(out, target.max(dim=-1)[1])

        losses.update(loss.item(), out.shape[0])
        accuracy.update(batch_acc, out.shape[0])
        precision.update(batch_precision)
        recall.update(batch_recall)
        f1.update(batch_f1)
        auroc.update(batch_auroc)

        # update confusion matrix
        _, targs = torch.max(target, 1)
        _, preds = torch.max(out, 1)
        for t, p in zip(targs.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        iter_time.update(time.time() - start)
        if idx % 5 == 0:
          print(('Epoch: [{0}][{1}/{2}]\t'
                 'Time: {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                 'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                 'Accur.: {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                 'Prec.: {precision.val:.4f} ({precision.avg:.4f})\t'
                 'Recall: {recall.val:.4f} ({recall.avg:.4f})\t'
                 'F1: {f1.val:.4f} ({f1.avg:.4f})\t'
                 'AUROC: {auroc.val:.4f} ({auroc.avg:.4f})').format(
                      epoch,
                      idx,
                      len(valid_loader),
                      iter_time=iter_time,
                      losses=losses,
                      accuracy=accuracy,
                      precision=precision,
                      recall=recall,
                      f1=f1,
                      auroc=auroc))
    
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("*Accur.: {accuracy.avg:.4f}\t"
          "Precision: {precision.avg:.4f}\t"
          "Recall: {recall.avg:.4f}\t"          
          "F1: {f1.avg:.4f}\t"
          "AUROC: {auroc.avg:.4f}".format(
            accuracy=accuracy, 
            precision=precision, 
            recall=recall, 
            f1=f1, 
            auroc=auroc))
    
    results = {
        'loss':losses.avg,
        'accuracy':accuracy.avg,
        'precision':precision.avg,
        'recall':recall.avg,
        'f1':f1.avg,
        'auroc':auroc.avg,
        'cm':cm
    }

    return results

def train_model(params, model, lv, num_class, tune_subset, current_dir, df_results=None, run_id=0,stop_early = False):
    os.chdir(current_dir)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_size = params['batch_size']
    lr = params['learning_rate']
    warmup = params['warmup']
    steps = params['steps']
    betas = params['betas']
    weight_decay = params['weight_decay']

    train_dataloader, valid_dataloader, _ = create_class_dataloaders(lv.source_class, lv.target_class, num_class=num_class, batch_size=batch_size, tune_subset=tune_subset, full_train=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    loss_type = params['loss_type']
    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "FL":
        fl_beta = params['fl_beta']
        fl_gamma = params['fl_gamma']
        
        train_index = train_dataloader.dataset.dataset.train_index
        cls_num_list = list(train_dataloader.dataset.dataset.data_df.filter(items=train_index, axis=0).finding_labels.value_counts().sort_index())
        per_cls_weights = reweight(cls_num_list, beta=fl_beta)
        per_cls_weights = per_cls_weights.to(device)
            
        criterion = FocalLoss(weight=per_cls_weights, gamma=fl_gamma)
    else:
        per_cls_weights = None

    train_results = {
        'loss':[],
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'auroc':[]
    }

    valid_results = {
        'loss':[],
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'auroc':[],
        'cm':None
    }

    stop_early_results = []

    if df_results is None:
        df_results = pd.DataFrame()
    
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr, warmup, steps)

        # train loop
        epoch_train_results = train(epoch, train_dataloader, model, optimizer, criterion, stop_early)

        # validation loop
        epoch_valid_results = validate(epoch, valid_dataloader, num_class, model, criterion)

        # Save the epochs results
        train_results['loss'].append(epoch_train_results['loss'])
        valid_results['loss'].append(epoch_valid_results['loss'])
        stop_early_results.append(epoch_valid_results['loss'])

        for metric in list(epoch_train_results.keys())[1:]:
            train_results[metric].append(epoch_train_results[metric].tolist())
            valid_results[metric].append(epoch_valid_results[metric].tolist())

            idx = (run_id * num_epochs) + epoch
            df_temp = create_temp_df(params, idx)
            df_temp['run_id'] = run_id
            df_temp['epoch'] = epoch
            df_temp['train_' + metric] = epoch_train_results[metric].tolist()
            df_temp['valid_' + metric] = epoch_valid_results[metric].tolist()

            df_results = pd.concat([df_results, df_temp])
        
        valid_results['cm'] = epoch_valid_results['cm']

        delete_files(folder='/logging/', file='results')
        df_results.to_csv('./logging/' + model.model_name + '_results.csv')
        
        if stop_early and epoch > 2: # Stop early if no improvement over last 3 epochs
            if stop_early_results[-1] > stop_early_results[-2] and stop_early_results[-2] > stop_early_results[-3] and stop_early_results[-3] > stop_early_results[-4]:
                break

    return model, train_results, valid_results, df_results

def train_model_optimize(params, model_type, lv, num_class, tune_subset, current_dir):
    os.chdir(current_dir)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dropout = params['dropout']
    batch_size = params['batch_size']
    lr = params['learning_rate']
    warmup = params['warmup']
    steps = params['steps']
    betas = params['betas']
    weight_decay = params['weight_decay']

    if model_type == "VGG":
        model = VGG(num_class=num_class, dropout=dropout)
        filename = "vgg"
    elif model_type == "EfficientNet":
        model = EfficientNet(num_class=num_class, dropout=dropout)
        filename = "efficientnet"
    model = model.to(device)

    train_dataloader, valid_dataloader, _ = create_class_dataloaders(lv.source_class, lv.target_class, num_class=num_class, batch_size=batch_size, tune_subset=tune_subset, full_train=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    loss_type = params['loss_type']
    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "BCE":
        criterion == nn.BCEWithLogitsLoss()
    elif loss_type == "FL":
        fl_beta = params['fl_beta']
        fl_gamma = params['fl_gamma']
        
        train_index = train_dataloader.dataset.dataset.train_index
        cls_num_list = list(train_dataloader.dataset.dataset.data_df.filter(items=train_index, axis=0).finding_labels.value_counts().sort_index())
        per_cls_weights = reweight(cls_num_list, beta=fl_beta)
        per_cls_weights = per_cls_weights.to(device)

        criterion = FocalLoss(weight=per_cls_weights, gamma=fl_gamma)
    else:
        per_cls_weights = None
    
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr, warmup, steps)

        # train loop
        train_results = train(epoch, train_dataloader, model, optimizer, criterion)

        # validation loop
        valid_results = validate(epoch, valid_dataloader, num_class, model, criterion)

        # TODO: checkpoint state dict, not python object

        checkpoint_data = {
            "epoch": epoch,
            # "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        if (epoch+1)%5 == 0:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                    pickle.dump(checkpoint_data, fp)
                checkpoint = Checkpoint.from_directory(checkpoint_dir)

                session.report(
                    {"loss": valid_results['loss'], 
                    "accuracy": valid_results['accuracy'].tolist(),
                    "precision": valid_results['precision'].tolist(),
                    "recall": valid_results['recall'].tolist(),
                    "f1": valid_results['f1'].tolist(),
                    "auroc": valid_results['auroc'].tolist()},
                    checkpoint=checkpoint
                )

            # report(metrics = {"loss": val_loss, "accuracy": val_acc},
            #        checkpoint=checkpoint)
            
    print("Training run finished.")

def train_model_full(params, model, lv, num_class, tune_subset, current_dir, df_results=None, run_id=0):
    os.chdir(current_dir)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_size = params['batch_size']
    lr = params['learning_rate']
    warmup = params['warmup']
    steps = params['steps']
    betas = params['betas']
    weight_decay = params['weight_decay']

    train_dataloader, _, _ = create_class_dataloaders(lv.source_class, lv.target_class, num_class=num_class, batch_size=batch_size, tune_subset=False, full_train=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    loss_type = params['loss_type']
    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "FL":
        fl_beta = params['fl_beta']
        fl_gamma = params['fl_gamma']
        
        train_index = train_dataloader.dataset.dataset.train_index
        cls_num_list = list(train_dataloader.dataset.dataset.data_df.filter(items=train_index, axis=0).finding_labels.value_counts().sort_index())
        per_cls_weights = reweight(cls_num_list, beta=fl_beta)
        per_cls_weights = per_cls_weights.to(device)
            
        criterion = FocalLoss(weight=per_cls_weights, gamma=fl_gamma)
    else:
        per_cls_weights = None

    train_results = {
        'loss':[],
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'auroc':[]
    }

    if df_results is None:
        df_results = pd.DataFrame()
    
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, lr, warmup, steps)

        # train loop
        epoch_train_results = train(epoch, train_dataloader, model, optimizer, criterion)

        # Save the epochs results
        train_results['loss'].append(epoch_train_results['loss'])

        for metric in list(epoch_train_results.keys())[1:]:
            train_results[metric].append(epoch_train_results[metric].tolist())

            idx = (run_id * num_epochs) + epoch
            df_temp = create_temp_df(params, idx)
            df_temp['run_id'] = run_id
            df_temp['epoch'] = epoch
            df_temp['train_' + metric] = epoch_train_results[metric].tolist()

            df_results = pd.concat([df_results, df_temp])

        delete_files(folder='/logging/', file='results')
        df_results.to_csv('./logging/' + model.model_name + '_results.csv')

    return model, train_results, df_results

def plot_curves(train_history, valid_history, filename, metric):
    '''
    Plot learning curves with matplotlib. Make sure training perplexity and validation perplexity are plot in the same figure
    :param train_history: training accuracy history of epochs
    :param valid_history: validation accuracy history of epochs
    :param metric: metric to plot
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation accuracy            #
    #    2) Plot learning curves of training and validation loss                #
    #############################################################################
    #plt.ioff()
    
    epochs = range(len(train_history))
    
    fig1 = plt.figure('Figure 1', figsize = (10, 8))
    plt.plot(epochs, train_history, label='train')
    plt.plot(epochs, valid_history, label='valid')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.title(filename + ' - ' + metric)
    plt.savefig('./plots/' + filename + '_' + metric + '.png')
    plt.close()
    #plt.show()