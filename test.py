# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:53:40 2024

@author: Feiyang Wu
"""
import os
import sys
import torch
import argparse
import warnings
import math
import importlib.util
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

def parse_args():
    parser = argparse.ArgumentParser(description='This is a demo for preparing dataset from scratch.')
    parser.add_argument('--config', type=str, help='Config file of the model.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Choose device for computing. CPU will be used if cuda is not available')
    parser.add_argument('--std', help='The standard token jdson file. Please given the root name of the file.')
    parser.add_argument('--checkpoint', help='Input the checkpoint file that end with ".pth".')
    parser.add_argument('--dist', default=None, help='Plot the distribution of prediction on train and valid data.')
    args = parser.parse_args()
    return args

def benchmark(model, dataset, dataset_original, parameter, device, dist):
    model.eval()  # Set model to evaluation mode
    indices = list(range(len(dataset)))
    split = math.floor(len(dataset)*0.8)
    
    train_indices = indices[:split]
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=parameter['train_cfg']['batch'], shuffle=True)
    
    valid_indices = indices[split:]
    valid_dataset = Subset(dataset, valid_indices)
    valid_loader = DataLoader(valid_dataset, batch_size=parameter['valid_cfg']['batch'], shuffle=True)
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    num_pos = 0
    num_neg = 0

    for i in range(len(dataset)):
        if int(dataset[i][1]) == 1:
            num_pos += 1
        else:
            num_neg += 1
    with torch.no_grad():  # Disable gradient computation
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
    
            # Reshape inputs to (seq_len, batch_size, input_dim)
            inputs = inputs.permute(2, 0, 1)  # Shape: (2200, batch_size, 18)
    
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
    
            # Apply threshold to get predicted classes
            predicted = (outputs >= 0.5).float()
    
            # Update counts
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            true_positive += ((predicted == 1) & (targets == 1)).sum().item()
            true_negative += ((predicted == 0) & (targets == 0)).sum().item()
            false_positive += ((predicted == 1) & (targets == 0)).sum().item()
            false_negative += ((predicted == 0) & (targets == 1)).sum().item()
            
            targets = targets.cpu()
            outputs = outputs.cpu()
        train_acc = train_correct / train_total

        valid_correct = 0
        valid_total = 0
        for inputs, targets in valid_loader:
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)
    
            # Reshape inputs to (seq_len, batch_size, input_dim)
            inputs = inputs.permute(2, 0, 1)  # Shape: (2200, batch_size, 18)
    
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
    
            # Apply threshold to get predicted classes
            predicted = (outputs >= 0.5).float()
    
            # Update counts
            valid_total += targets.size(0)
            valid_correct += (predicted == targets).sum().item()
            
            true_positive += ((predicted == 1) & (targets == 1)).sum().item()
            true_negative += ((predicted == 0) & (targets == 0)).sum().item()
            false_positive += ((predicted == 1) & (targets == 0)).sum().item()
            false_negative += ((predicted == 0) & (targets == 1)).sum().item()

        valid_acc = valid_correct / valid_total
        
    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    positive_correct_rate = true_positive/(true_positive+false_negative)
    neagtive_correct_rate = true_negative/(true_negative+false_positive)
    acc = (train_correct + valid_correct) / (train_total + valid_total)
    
    # Calculate F1 score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    rate =  {'# of 13':num_pos, '# of 14':num_neg,
             'train_acc':train_acc,
             'valid_acc':valid_acc,
             'acc':acc, 'f1':f1, 'positive_correct_rate':positive_correct_rate, 'neagtive_correct_rate':neagtive_correct_rate,
             'true_positive':true_positive, 'true_negative':true_negative, 'false_positive':false_positive, 'false_negative':false_negative}
    
    for key in rate:
        if rate[key] < 1:
            print(f'{key}: {rate[key] * 100:.2f}%')
        else:
            print(f'{key}: {rate[key]}')
    
    truth = []
    pred = []
    this = dataset_original
    for i in range(len(this)):
        input, target = this[i]
        input = input.unsqueeze(1).to(device)
        target = float(target)
        input = input.permute(2,1,0)
        output = float(model(input).squeeze())
        pred.append(output)
        truth.append(this.get_rate(i))
            
    if dist != None:
        plt.figure(figsize=(13, 5))
        plt.scatter(truth,pred, s=0.5)
        plt.hlines(y=0.5, xmin=min(truth)-0.1, xmax=max(truth)+0.1, colors='red', linestyles='dashed')
        plt.vlines(x=13.95, ymin=0, ymax=1, colors='red', linestyles='dashed')
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
        plt.xlim(min(truth)-0.1, max(truth)+0.1)
        plt.ylim(0, 1)
        plt.xlabel("Rating")
        plt.ylabel("Probability of being 13 or 13+")
        plt.title('Probability of all samples (no duplicates)')
        plt.savefig(os.getcwd() + dist + '/distribution.png')
        print('Probability graph plotted and saved.')
            
            
    return rate




def main():
    warnings.filterwarnings("ignore")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    args = parse_args()
    # Get the current working directory
    current_working_directory = os.getcwd()
    sys.path.append(current_working_directory+'/tools/')
    sys.path.append(current_working_directory+'/model/')
    from ChartStats import chartStats
    from TAMAMo import TokenAlignedMaimaiAnalyzerModel
    # Path to the file
    file_path = args.config

    # Get the root name
    root_name = Path(file_path).stem

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(root_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[root_name] = module
    spec.loader.exec_module(module)
    parameter = module.parameter
    
    
    # Setting up dataset
    print('Start loading dataset...')
    dataset = chartStats( current_working_directory + parameter['dataset_cfg']['path'], parameter['dataset_cfg']['boundary'])
    dataset_original = chartStats( current_working_directory + '/std_tokens_lib/' + args.std, parameter['dataset_cfg']['boundary'])
    print(f'Dataset loaded. {len(dataset)} samples in total.')
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    
    # Setting up model and training details
    model = TokenAlignedMaimaiAnalyzerModel(nhead = parameter['model_cfg']['nhead'],
                                            hidden_dim = parameter['model_cfg']['hidden_dim'], 
                                            num_layers = parameter['model_cfg']['num_layers'],
                                            hidden_neuron = parameter['model_cfg']['hidden_neuron'], 
                                            max_len = parameter['model_cfg']['max_len']).to(device)
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint)
    
    benchmark(model, dataset, dataset_original, parameter, device, args.dist)
        
if __name__ == '__main__':
    main()
    

    