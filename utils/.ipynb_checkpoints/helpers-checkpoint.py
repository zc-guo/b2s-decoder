## Modified from nn_model_helper_funcs.py

import os, sys
import numpy as np
from shutil import copyfile
from scipy.stats import pearsonr

def setup(args):
    """
    Set up exp folder.
    """
    print('='*8 + ' Prepare for training ' + '='*8)
    config_path = args.conf_dir
    exp_path = args.exp_dir

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(f"Folder {exp_path} doesn't exist. Create it.")
    
    results_path = os.path.join(exp_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Folder {results_path} doesn't exist. Create it.")

    trained_models_path = os.path.join(exp_path, 'trained_models')
    if not os.path.exists(trained_models_path):
        os.makedirs(trained_models_path)
        print(f"Folder {trained_models_path} doesn't exist. Create it.")
        
    conf_dst = os.path.join(exp_path, os.path.basename(config_path))        
    copyfile(config_path, conf_dst)
    print(f'Copied {os.path.basename(config_path)} to {conf_dst}.')
    
    model_dst = os.path.join(exp_path, 'models.py')
    copyfile('utils/models.py', model_dst)
    print(f'Copied utils/models.py to {model_dst}.')
    print('Setup completed.\n')
    

def prepare_data(data, out_path, subject = 1):
    """
    Prepare the data for training the model. Specifically, extract the data from the subject for which decoding will be performed and put them in the right format.
    
        data: EEG and envelope data (.npz object)
        out_path: output path for the data of this subject
        subject: ID of subject to perform decoding (default: 1)
    """
    # Load and reshape
    X, Y = data["eegs"], data["envs"]
    X = np.transpose(X, (0, 2, 1))
    Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
    
    # Get subject ID and phoneme lists
    sub_list = data["subject"]
    ipa_list = data["ipa"]
    
    # List of unique phonemes
    phoneme_list = np.unique(ipa_list)
    
    # Prepare the data
    # Extract the EEGs and envelopes for this subject. Do this separately for each phoneme
    datasets_by_phoneme = {}
    for phoneme in phoneme_list:
        datasets_by_phoneme[phoneme] = {}
        datasets_by_phoneme[phoneme]["X"] = X[np.all([ipa_list == phoneme, sub_list == subject], axis = 0), :, :]
        datasets_by_phoneme[phoneme]["Y"] = Y[np.all([ipa_list == phoneme, sub_list == subject], axis = 0), :, :]
    
    # Create output folder for this subject
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # Check the shape of the dataframe of each phoneme and write it to a data summary
    with open(os.path.join(out_path, 'data_summary.txt'), 'w') as ds:
        for p in datasets_by_phoneme.keys():
            print("Phoneme:", p, file = ds)
            print("EEGs (X) shape:", datasets_by_phoneme[p]["X"].shape, file = ds)
            print("Envelopes (Y) shape:", datasets_by_phoneme[p]["Y"].shape, file = ds)
            print("-------", file = ds)
    
    return datasets_by_phoneme
    
def pearsonr_batch(preds, targets):
    """
    Calculate Pearon's r in batch.
        preds: predicted sequences
        targets: target (ground-truth) sequences
    """
    rs = []
    for i in range(preds.size(0)):
        [r, p] = pearsonr(np.squeeze(preds[i]), np.squeeze(targets[i]))
        rs.append(r)
    return rs

def mse_batch(preds, targets):
    """
    Calculate MSE in batch.
        preds: predicted sequences
        targets: target (ground-truth) sequence
    """
    mses = []
    for i in range(preds.size(0)):
        pred_tmp = np.array(np.squeeze(preds[i]))
        target_tmp = np.array(np.squeeze(targets[i]))
        mse = np.mean((pred_tmp - target_tmp) ** 2)
        mses.append(mse)
    return mses 