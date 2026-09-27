import argparse
import copy
import csv

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
from training.focal_loss import FocalLoss, reweight

from data.dataloaders import create_class_dataloaders
from configs.env_setup import LocalVariables
from configs.utils import delete_files, train, validate, train_model, train_model_optimize, train_model_full, adjust_learning_rate, plot_curves
from configs.train_config import create_tuning_params

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import multiprocessing

from functools import partial

import glob

from captum.attr import IntegratedGradients, Saliency, LayerGradCam, GuidedBackprop, LayerAttribution, GuidedGradCam, LayerActivation
from captum.attr import visualization as viz

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="VGG")
    parser.add_argument("--run-type", type=str, default="visualize")
    parser.add_argument("--binary-class", type=str, default='True')
    parser.add_argument("--tune-subset", type=str, default='False')
    parsed_args = parser.parse_args()
    return parsed_args

def main():

    # TODO: produce visualization of layer results

    parsed_args = parse_args()
    if parsed_args.tune_subset == 'True':
        tune_subset = True
    elif parsed_args.tune_subset == 'False':
        tune_subset = False
    if parsed_args.binary_class == 'True':
        num_class = 2
    elif parsed_args.binary_class == 'False':
        num_class = 15

    lv = LocalVariables("./configs/local_config.yaml")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # seed = np.random.randint(low=1, high=500)
    # shuffle = True

    # sample, target = next(iter(train_dataloader))
    # sample = sample.to('cuda')
    # model.forward(sample[0:2,:,:,:])
    params = create_tuning_params(run_type=parsed_args.run_type)
    num_runs = params['num_runs']

    if parsed_args.run_type == "single":
        params = create_tuning_params(run_type=parsed_args.run_type)

        if parsed_args.model_type == "VGG":
            model = VGG(num_class=num_class, dropout=params['input']['dropout'])
            filename = "vgg"
        elif parsed_args.model_type == "EfficientNet":
            model = EfficientNet(num_class=num_class, dropout=params['input']['dropout'])
            filename = "efficientnet"
        model = model.to(device)

        model, train_results, valid_results, _ = train_model(params=params['input'], model=model, lv=lv, num_class=num_class, tune_subset=tune_subset, current_dir=current_dir, df_results=None, run_id=0,
                                                             stop_early = True)

        final_results = {
            'loss':99999,
            'accuracy':0.0,
            'precision':0.0,
            'recall':0.0,
            'f1':0.0,
            'auroc':0.0
        }
        
        for metric in train_results.keys():
            plot_curves(train_results[metric], valid_results[metric], filename, metric)
            final_results[metric] = valid_results[metric][-1]
            #df_results['train_' + metric] = train_results[metric][-1]
            #df_results['valid_' + metric] = valid_results[metric][-1]

        print(f"Accur.: {final_results['accuracy']:.4f}\t Precision: {final_results['precision']:.4f}\t Recall: {final_results['recall']:.4f}\t F1: {final_results['f1']:.4f}\t AUROC: {final_results['auroc']:.4f}")

        final_cm = valid_results['cm']
        per_cls_acc = final_cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
    
        final_cm = final_cm.numpy()
        np.savetxt('./logging/' + filename + '_cm.csv', final_cm, delimiter=',')
        
        delete_files(folder="/checkpoints/", file="epoch")
        torch.save(model.state_dict(), "./checkpoints/" + filename + "_final.pth")
    
    elif parsed_args.run_type == 'full':
        params = create_tuning_params(run_type=parsed_args.run_type)

        if parsed_args.model_type == "VGG":
            model = VGG(num_class=num_class, dropout=params['input']['dropout'])
            filename = "vgg"
        elif parsed_args.model_type == "EfficientNet":
            model = EfficientNet(num_class=num_class, dropout=params['input']['dropout'])
            filename = "efficientnet"
        model = model.to(device)

        model, train_results, _ = train_model_full(params=params['input'], model=model, lv=lv, num_class=num_class, tune_subset=tune_subset, current_dir=current_dir, df_results=None, run_id=0)

        final_results = {
            'loss':99999,
            'accuracy':0.0,
            'precision':0.0,
            'recall':0.0,
            'f1':0.0,
            'auroc':0.0
        }
        
        for metric in train_results.keys():
            final_results[metric] = train_results[metric][-1]
            #plot_curves(train_results[metric], valid_results[metric], filename, metric)

        print(f"Accur.: {final_results['accuracy']:.4f}\t Precision: {final_results['precision']:.4f}\t Recall: {final_results['recall']:.4f}\t F1: {final_results['f1']:.4f}\t AUROC: {final_results['auroc']:.4f}")
        
        delete_files(folder="/checkpoints/", file="epoch")
        torch.save(model.state_dict(), "./checkpoints/" + filename + "_final.pth")

    elif parsed_args.run_type == 'random':

        best_results = {
            'run_id':99999,
            'loss':99999,
            'accuracy':0.0,
            'precision':0.0,
            'recall':0.0,
            'f1':0.0,
            'auroc':0.0,     
        }

        df_results = pd.DataFrame()
        
        for run_id in range(num_runs):
            print("-----------------------------------")
            print('Trial: ' + str(run_id))
        
            params = create_tuning_params(run_type=parsed_args.run_type, seed=run_id*2)

            if parsed_args.model_type == "VGG":
                model = VGG(num_class=num_class, dropout=params['tune']['dropout'])
                filename = "vgg"
            elif parsed_args.model_type == "EfficientNet":
                model = EfficientNet(num_class=num_class, dropout=params['tune']['dropout'])
                filename = "efficientnet"
            model = model.to(device)

            model, train_results, valid_results, df_results = train_model(params=params['tune'], model=model, lv=lv, num_class=num_class, tune_subset=tune_subset, current_dir=current_dir, df_results=df_results, run_id=run_id)

            fn = filename + '_run' + str(run_id)
            final_results = {
                'loss':99999,
                'accuracy':0.0,
                'precision':0.0,
                'recall':0.0,
                'f1':0.0,
                'auroc':0.0
            }

            for metric in train_results.keys():
                final_results[metric] = valid_results[metric][-1]
                #df_results['train_' + metric] = train_results[metric][-1]
                #df_results['valid_' + metric] = valid_results[metric][-1]
                plot_curves(train_results[metric], valid_results[metric], fn, metric)

            print(f"Accur.: {final_results['accuracy']:.4f}\t Precision: {final_results['precision']:.4f}\t Recall: {final_results['recall']:.4f}\t F1: {final_results['f1']:.4f}\t AUROC: {final_results['auroc']:.4f}")

            final_cm = valid_results['cm']
            per_cls_acc = final_cm.diag().detach().numpy().tolist()
            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))  
            final_cm = final_cm.numpy()
            np.savetxt('./logging/' + fn + '_cm.csv', final_cm, delimiter=',')

            delete_files(folder="/checkpoints/", file="epoch")
            torch.save(model.state_dict(), "./checkpoints/" + fn + "_final.pth")
            
            if (num_class != 2 and final_results['f1'] > best_results['f1']) or (num_class == 2 and final_results['accuracy'] > best_results['accuracy']):
                best_results['run_id'] = run_id
                best_results['loss'] = final_results['loss']
                best_results['accuracy'] = final_results['accuracy']
                best_results['precision'] = final_results['precision']
                best_results['recall'] = final_results['recall']
                best_results['f1'] = final_results['f1']
                best_results['auroc'] = final_results['auroc'] 
                
                delete_files(folder="/logging/best/")
                df_best = pd.DataFrame(best_results, index=[0])
                df_best.to_csv('./logging/best/' + fn + '_best_results.csv')
                best_cm = final_cm.copy()
                np.savetxt('./logging/best/' + fn + '_best_cm.csv', best_cm, delimiter=',')
                
                best_model = copy.deepcopy(model)
                delete_files(folder="/checkpoints/best/")
                torch.save(best_model.state_dict(), "./checkpoints/best/" + fn + "_best_model.pth")

    elif parsed_args.run_type == "optimize":
        model_type = parsed_args.model_type

        scheduler = ASHAScheduler(
            metric="f1",
            mode="max"
        )
        
        tuning_result = tune.run(
            tune.with_parameters(train_model_optimize, model_type=model_type, lv=lv, num_class=num_class, tune_subset=tune_subset, current_dir=current_dir),
            resources_per_trial={'cpu':multiprocessing.cpu_count(), 'gpu':torch.cuda.device_count()}, 
            config=params['tune'],
            num_samples = params['num_runs'],
            scheduler=scheduler,
            storage_path = current_dir + '/logging'
        )

        best_trial = tuning_result.get_best_trial("f1", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
        print(f"Best trial final validation precision: {best_trial.last_result['precision']}")
        print(f"Best trial final validation recall: {best_trial.last_result['recall']}")
        print(f"Best trial final validation f1: {best_trial.last_result['f1']}")
        print(f"Best trial final validation auroc: {best_trial.last_result['auroc']}")

    elif parsed_args.run_type == "test":
        if parsed_args.model_type == "VGG":
            model = VGG(num_class=num_class, dropout=params['input']['dropout'])
            filename = "vgg"
        elif parsed_args.model_type == "EfficientNet":
            model = EfficientNet(num_class=num_class, dropout=params['input']['dropout'])
            filename = "efficientnet"
        model = model.to(device)

        checkpoint = torch.load(glob.glob('.\\checkpoints\\best_full\\*.pth')[0])

        model.load_state_dict(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        _, _, test_dataloader = create_class_dataloaders(lv.source_class, lv.target_class, num_class=num_class, batch_size=params['input']['batch_size'], tune_subset=False, full_train=True)

        loss_type = params['input']['loss_type']
        if loss_type == "CE":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "BCE":
            criterion == nn.BCEWithLogitsLoss()

        test_results = validate(epoch = 0, valid_loader = test_dataloader, num_class = num_class, model = model, criterion = criterion)

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auroc']:
            test_results[metric] = test_results[metric].cpu().numpy().tolist()

        test_cm = test_results.pop('cm')
        df_test = pd.DataFrame(test_results, index=[0])
        df_test.to_csv('.\\checkpoints\\best_full\\test_results.csv')
        np.savetxt('.\\checkpoints\\best_full\\best_cm.csv', test_cm, delimiter=',')

    elif parsed_args.run_type == "visualize":
        if parsed_args.model_type == "VGG":
            model = VGG(num_class=num_class, dropout=params['input']['dropout'])
            filename = "vgg"
        elif parsed_args.model_type == "EfficientNet":
            model = EfficientNet(num_class=num_class, dropout=params['input']['dropout'])
            filename = "efficientnet"

        checkpoint = torch.load(glob.glob('.\\checkpoints\\best_full\\*.pth')[0])

        model.load_state_dict(checkpoint)
        model.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        device = 'cpu'
        model = model.to(device)

        train_dataloader, _, test_dataloader = create_class_dataloaders(lv.source_class, lv.target_class, num_class=num_class, batch_size=params['input']['batch_size'], tune_subset=False, full_train=True)

        # average_class_images = torch.zeros([num_class]+list(train_dataloader.dataset[0][0].shape))
        # target_counts = torch.zeros(num_class)

        # for img_num, (img,target) in enumerate(train_dataloader):
        #     which_target = target.argmax(dim = 1)
        #     for target_cat in range(num_class):
        #         this_target = which_target == target_cat
        #         average_class_images[target_cat,:,:,:] += img[this_target,:,:,:].sum(dim=0)
        #         target_counts[target_cat] += this_target.sum().tolist()
        #     print(img_num/len(train_dataloader))
        
        # for target_num in range(num_class):
        #     average_class_images[target_num,:,:,:] /= target_counts[target_num]

        saliency = Saliency(model)
        # gradcam = LayerGradCam(model, layer = model.cnn[4])
        gradcam = LayerGradCam(model, layer = model.pretrained_layer[19])
        labels = ["Clear","Disease"]
        train_data_length = len(train_dataloader.dataset)
        test_data_length = len(test_dataloader.dataset)

        for img_num, (img,target) in enumerate(train_dataloader.dataset):
            img = img.unsqueeze(0)
            prediction = labels[model(img).argmax()]
            img.requires_grad = True
            target = target.argmax()
            actual = labels[target]
            img = img.to(device)
            target = target.to(device)
            img_index = train_dataloader.dataset.dataset.data_df.iloc[img_num]['img_index']
            attr_sm = saliency.attribute(img, target)
            attr_sm = torch.permute(attr_sm,(0,2,3,1)).squeeze()
            attr_gc = gradcam.attribute(img,target)
            attr_gc = LayerAttribution.interpolate(attr_gc, img.shape[2:])
            img = torch.permute(img,(0,2,3,1))
            img = img.squeeze().cpu().detach().numpy()
            attr_gc = torch.permute(attr_gc,(0,2,3,1))
            attr_gc = attr_gc[0].cpu().detach().numpy()
            attr_sm = attr_sm.cpu().detach().numpy()

            pos_gc_f, pos_gc_ax = viz.visualize_image_attr(attr_gc, img, method = "blended_heat_map", sign = "positive", 
                                         title = f"Positive Gradcam, Predicted: {prediction}, Actual: {actual}",
                                         use_pyplot=False)
            neg_gc_f, neg_gc_ax = viz.visualize_image_attr(attr_gc, img, method = "blended_heat_map", sign = "negative", 
                                         title = f"Negative Gradcam, Predicted: {prediction}, Actual: {actual}",
                                         use_pyplot=False)
            sm_f, sm_ax = viz.visualize_image_attr(attr_sm,use_pyplot=False)

            img_index = img_index.replace('.png','')

            pos_gc_f.savefig(f'visualizations/gradcam/{img_index}_pos.png')
            neg_gc_f.savefig(f'visualizations/gradcam/{img_index}_neg.png')
            sm_f.savefig(f'visualizations/saliency/{img_index}.png')

            print(f'Saved visualizations for {img_num+1} out of {train_data_length} training images')

        print('Training visualizations complete.')

        for img_num, (img,target) in enumerate(test_dataloader.dataset):
            img = img.unsqueeze(0)
            prediction = labels[model(img).argmax()]
            img.requires_grad = True
            target = target.argmax()
            actual = labels[target]
            img = img.to(device)
            target = target.to(device)
            img_index = test_dataloader.dataset.dataset.data_df.iloc[img_num]['img_index']
            attr_sm = saliency.attribute(img, target)
            attr_sm = torch.permute(attr_sm,(0,2,3,1)).squeeze()
            attr_gc = gradcam.attribute(img,target)
            attr_gc = LayerAttribution.interpolate(attr_gc, img.shape[2:])
            img = torch.permute(img,(0,2,3,1))
            img = img.squeeze().cpu().detach().numpy()
            attr_gc = torch.permute(attr_gc,(0,2,3,1))
            attr_gc = attr_gc[0].cpu().detach().numpy()
            attr_sm = attr_sm.cpu().detach().numpy()

            pos_gc_f, pos_gc_ax = viz.visualize_image_attr(attr_gc, img, method = "blended_heat_map", sign = "positive", 
                                         title = f"Positive Gradcam, Predicted: {prediction}, Actual: {actual}",
                                         use_pyplot=False)
            neg_gc_f, neg_gc_ax = viz.visualize_image_attr(attr_gc, img, method = "blended_heat_map", sign = "negative", 
                                         title = f"Negative Gradcam, Predicted: {prediction}, Actual: {actual}",
                                         use_pyplot=False)
            sm_f, sm_ax = viz.visualize_image_attr(attr_sm,use_pyplot=False)

            img_index = img_index.replace('.png','')

            pos_gc_f.savefig(f'visualizations/gradcam/{img_index}_pos.png')
            neg_gc_f.savefig(f'visualizations/gradcam/{img_index}_neg.png')
            sm_f.savefig(f'visualizations/saliency/{img_index}.png')

            print(f'Saved visualizations for {img_num+1} out of {test_data_length} test images')

        print('Test visualizations complete.')

if __name__ == '__main__':
    main()

