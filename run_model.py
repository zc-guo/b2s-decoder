import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from utils.helpers import setup, prepare_data, pearsonr_batch, mse_batch
from utils.models import LSTM_basic
from shutil import copyfile

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

def cross_validate_nn(dataset, args):
    """
    Performs k-fold cross-validation using the imported neural network model.
        dataset: EEG and envelope data (returned by prepare_data)   
    """
    # Load training and model configurations
    config = yaml.load(open(args.conf_dir, 'r'), Loader = yaml.FullLoader)
    k = config['Training_setup']['n_folds']
    batch_size = config['Training_setup']['batch_size']
    learning_rate = config['Training_setup']['learning_rate']
    n_epochs = config['Training_setup']['epochs']
    n_channels = config['Data_setup']['n_channels']
    rnn_dim = config['NN_setup']['rnn_dim']
    n_rnn_layers = config['NN_setup']['n_rnn_layers']
    dropout = config['NN_setup']['dropout']

    # Check GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
        print('GPUs not available, use CPU')
    
    # Get the EEGs (X) and envelopes (Y) of the training data
    X = dataset['X']
    Y = dataset['Y']
    
    # Initiate dict for storing the results and models
    results = {}
    results['cv_cor'] = {}
    results['cv_mse'] = {}
    results['targets'] = []
    results['preds'] = []
    
    models = {}
    
    # K-fold cross-validation
    kf = KFold(n_splits = k) 
    
    for fold_i, (train_i, out_i) in enumerate(kf.split(X)):
        
        # Split data into train and held-out folds
        XFold_train, YFold_train = X[train_i, :, :], Y[train_i, :, :]
        XFold_out, YFold_out = X[out_i, :, :], Y[out_i, :, :]
        print(f'    Cross-validating fold {fold_i+1}...')
    
        # Conver to torch.Tensor        
        XFold_train = torch.Tensor(XFold_train)
        YFold_train = torch.Tensor(YFold_train)
        XFold_out = torch.Tensor(XFold_out)
        YFold_out = torch.Tensor(YFold_out)
        
        train_dataset = torch.utils.data.TensorDataset(XFold_train, YFold_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size = batch_size, 
                                                   shuffle = False)
        
        test_dataset = torch.utils.data.TensorDataset(XFold_out, YFold_out)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size = batch_size, 
                                                  shuffle = False)
        
        # Initiate the model
        model = LSTM_basic(n_channels = n_channels, hidden_size = rnn_dim, 
                           n_layers = n_rnn_layers, dropout = dropout).to(device)
        model = model.to(device)
        
        # Initialize optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        
        # Run for n_epochs
        for epoch in range(0, n_epochs):
            
            print("  ---- Starting epoch %d ----" % (epoch + 1), end = '\r')
            
            for i, (inputs, targets) in enumerate(train_loader, 0):
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()    
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                print('      Loss:%0.6f' % (loss.item()), end = '\r')
                
                loss.backward()
                optimizer.step()
                # scheduler.step()
                
        print(f'    Training completed for fold {fold_i+1}')
        
        # Save model to models
        models[str(fold_i+1)] = model.state_dict()
        
        # Initiate lists to keep track of Pearson rs and MSEs for the held-out set
        fold_rs = []
        fold_mses = []
        
        with torch.no_grad():
            
            # Iterate over the test data and generate predictions
            for i, (inputs, targets) in enumerate(test_loader, 0):
                
                # Send the inputs and targets to CPU or GPU
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Generate outputs
                outputs = model(inputs)
                
                # Compute Pearson rs and MSEs and save the results
                rs = pearsonr_batch(outputs.cpu(), targets.cpu())
                mses = mse_batch(outputs.cpu(), targets.cpu())
                
                fold_rs.extend(rs)
                fold_mses.extend(mses)
                results['targets'].extend([np.squeeze(x) for x in targets])
                results['preds'].extend([np.squeeze(x) for x in outputs])
                
            # Print results
            print('    Average Pearson r for fold %d: %0.6f' % (fold_i+1, np.mean(fold_rs)))
            print('    Average MSE for fold %d: %0.8f' % (fold_i+1, np.mean(fold_mses)))
            print('    --------------------------------')
        
        # Record results
        results['cv_cor'][str(fold_i+1)] = fold_rs
        results['cv_mse'][str(fold_i+1)] = fold_mses
    
    return results, models


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--exp_dir', default = 'exp')

    args = parser.parse_args()
    setup(args)
    
    config = yaml.load(open(args.conf_dir, 'r'), Loader = yaml.FullLoader)
    subjects = config['Data_setup']['subjects']
    data_path = config['Data_setup']['path']
    data_fn = config['Data_setup']['data_file']
    save_model = config['Testing_setup']['save_model']
    data = np.load(os.path.join(data_path, data_fn))
    print('Run models for the following subjects:', subjects, '\n')
    
    for sub in subjects:
        print(f'Preparing data and output folder for data summary...', end = ' ', flush = True)
        sub_out_path = os.path.join('exp', 'data_summary')          
        datasets_by_phoneme = prepare_data(data = data, subject = sub, out_path = sub_out_path)
        print('Finished\n')
        
        print('Now start training...')
        results = {}
        models = {}
        for p in datasets_by_phoneme.keys():
            print(f'==== Running cross-validation for {p} ====')
            results[p], models[p] = cross_validate_nn(datasets_by_phoneme[p], args)
            
        # Save the results and trained models (if needed)
        np.savez(f'exp/results/sub_{sub}.npz', results = results)
        
        if save_model:
            np.savez(f'exp/trained_models/sub_{sub}.npz', results = results)
        
        print(f'==== Training done for subject {sub}. Results saved. ====')
    
    print('All done!')
            
        
        
    
    