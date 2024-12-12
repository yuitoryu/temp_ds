import torch
import math
import importlib.util
import sys
import argparse
import os
import warnings
import logging
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
from matplotlib import pyplot as plt 
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file of the model.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Choose device for computing. CPU will be used if cuda is not available.')
    parser.add_argument('--valid', default=False, help='Do validation while training.')
    parser.add_argument('--lossplot', default=False, help='Plot your loss.')
    parser.add_argument('--save_dir', type=str, help='Directory for saving the model.')
    parser.add_argument('--dataset', default=None, help='Input this parameter if you don\'t want to use the dataset in dedicated place. Use relative path!')
    args = parser.parse_args()
    return args


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
    if args.dataset == None:
        dataset = chartStats( current_working_directory + parameter['dataset_cfg']['path'], parameter['dataset_cfg']['boundary'])
    else:
        dataset = chartStats( current_working_directory + '/' + args.dataset, parameter['dataset_cfg']['boundary'])
    print(f'Dataset loaded. {len(dataset)} samples in total.')
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    
    # Setting up model and training details
    model = TokenAlignedMaimaiAnalyzerModel(nhead = parameter['model_cfg']['nhead'],
                                            hidden_dim = parameter['model_cfg']['hidden_dim'], 
                                            num_layers = parameter['model_cfg']['num_layers'],
                                            hidden_neuron = parameter['model_cfg']['hidden_neuron'], 
                                            max_len = parameter['model_cfg']['max_len']).to(device)
    if parameter['model_cfg']['pretrained_from'] != None:
        checkpoint = torch.load(current_working_directory + parameter['model_cfg']['pretrained_from'], weights_only=True)
        model.load_state_dict(checkpoint)

    # Start training
    if not args.valid:
        train_loss = train(model, parameter['train_cfg'], dataset, args.save_dir, device, args.valid, parameter['valid_cfg'])
    else:
        train_loss, valid_loss = train(model, parameter['train_cfg'], dataset, args.save_dir, device, args.valid, parameter['valid_cfg'])
    
    epoch = parameter['train_cfg']['epoch']
    if args.lossplot:
        plt.figure(figsize=(13, 5))
        plt.plot(range(1,epoch+1), train_loss, label='train_loss')
        if args.valid:
            plt.plot(range(1,epoch+1), valid_loss, label='valid_loss')
        plt.xlabel("Epoch")
        plt.ylabel("Avearge loss")
        folder = str(Path(args.save_dir).parent)
        plt.legend()
        plt.savefig(folder + "/plot.png")
    
    

def train(model, train_cfg, dataset, save_dir, device, valid=False, valid_cfg=None):
    indices = list(range(len(dataset)))
    split = math.floor(len(dataset)*0.8)
    train_indices = indices[:split]
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch'], shuffle=True)
    if valid:
        valid_indices = indices[split:]
        valid_dataset = Subset(dataset, valid_indices)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_cfg['batch'], shuffle=True)
    
    
    log_file = Path(save_dir).parent / "training.log"
    logging.basicConfig(
        filename=log_file,
        filemode = 'w',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    base_lr = train_cfg['base_lr']
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = MultiStepLR(optimizer, 
                            milestones = train_cfg['milestone'], 
                            gamma = train_cfg['gamma'])
    num_epochs = train_cfg['epoch']
    folder = str(Path(save_dir).parent)
    
    logging.info("Start training process...")
    print('Start training process...')
    cur_min = math.inf
    cur_id = 0
    train_loss = []
    valid_loss = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            # Reshape inputs to (seq_len, batch_size, input_dim)
            inputs = inputs.permute(2, 0, 1)    # Shape: (2200, batch_size, 18)

            # Forward Pass
            outputs = model(inputs)
            #print(outputs.shape)
            outputs = outputs.squeeze(-1)

            # Compute Loss
            loss = criterion(outputs, targets)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        running_loss /= len(train_dataset)
        train_loss.append(running_loss)
        torch.save(model.state_dict(), folder + '/epoch'+str(epoch+1) + '.pth')
        
        if running_loss < cur_min:
            cur_id = epoch+1
            cur_min = running_loss
        
        if not valid:    
            log_message = f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}'
            print('=======================================================================================')
            print(log_message + f" at {datetime.now()}")
            logging.info(log_message)
            continue
        
        val_loss = 0.0
        model.eval()
        for inputs, targets in valid_loader:
            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            # Reshape inputs to (seq_len, batch_size, input_dim)
            inputs = inputs.permute(2, 0, 1)    # Shape: (2200, batch_size, 18)

            # Forward Pass
            outputs = model(inputs)
            #print(outputs.shape)
            outputs = outputs.squeeze(-1)
            
            # Compute Loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(valid_dataset)
        valid_loss.append(val_loss)
        
        log_message = f'Epoch [{epoch+1}/{num_epochs}], Train_loss: {running_loss:.4f}, Valid_loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}'
        print('=======================================================================================')
        print(log_message + f" at {datetime.now()}")
        logging.info(log_message)
        
    print(f'Best model comes from Epoch {cur_id}.')
    logging.info(f'Best model comes from Epoch {cur_id}.')
    print('Model training is done. Start saving checkpoint file...')
    logging.info('Model training is done. Start saving checkpoint file...')
    checkpoint = torch.load(folder +'/epoch'+str(cur_id) + '.pth', weights_only=True)
    model.load_state_dict(checkpoint)
    torch.save(model.state_dict(), save_dir)
    print('Checkpoint file saved.')
    logging.info('Checkpoint file saved.')
    
    if valid:
        return train_loss, valid_loss
    return train_loss
     
        
if __name__ == '__main__':
    main()
    