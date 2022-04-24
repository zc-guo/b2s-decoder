# b2sd_nn_model_helpers.py
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nn_models import LSTM_basic
from scipy.stats import pearsonr

def preprocess_data(train_dat, test_dat):
    
    # Preprocess: basically the same as in mTRF_model.ipynb
    XTrain, YTrain, XTest, YTest = train_dat["train_eegs"], train_dat["train_envs"], test_dat["test_eegs"], test_dat["test_envs"]
    
    XTrain = np.transpose(XTrain, (0, 2, 1))
    YTrain = np.reshape(YTrain, (YTrain.shape[0], YTrain.shape[1], 1))
    XTest = np.transpose(XTest, (0, 2, 1))
    YTest = np.reshape(YTest, (YTest.shape[0], YTest.shape[1], 1))
    
    Train_sub_list = train_dat["subject"]
    Test_sub_list = test_dat["subject"]
    subject = 1
    Train_ipa_list = train_dat["ipa"]
    Test_ipa_list = test_dat["ipa"]
    
    vowel_list = np.unique(Train_ipa_list)
    
    datasets_by_vowel = {}
    
    for vowel in vowel_list:
        # print(vowel)
        datasets_by_vowel[vowel] = {}
        XTrain_tmp = XTrain[np.all([Train_ipa_list == vowel, Train_sub_list == subject], axis = 0), :, :]
        YTrain_tmp = YTrain[np.all([Train_ipa_list == vowel, Train_sub_list == subject], axis = 0), :, :]
        XTest_tmp = XTest[np.all([Test_ipa_list == vowel, Test_sub_list == subject], axis = 0), :, :]
        YTest_tmp = YTest[np.all([Test_ipa_list == vowel, Test_sub_list == subject], axis = 0), :, :]
        
        ## Combine the training and test sets, since I'll perform 10-fold cross-validation
        datasets_by_vowel[vowel]["XTrain"] = np.concatenate((XTrain_tmp, XTest_tmp), axis = 0)
        datasets_by_vowel[vowel]["YTrain"] = np.concatenate((YTrain_tmp, YTest_tmp), axis = 0)
        
    # Check the shape of each dataframe
    for v in datasets_by_vowel.keys():
        print("Vowel:", v)
        print("XTrain shape:", datasets_by_vowel[v]["XTrain"].shape)
        print("YTrain shape:", datasets_by_vowel[v]["YTrain"].shape)
        print("-------")
    
    return datasets_by_vowel
    
def pearsonr_batch(preds, targets):
    rs = []
    for i in range(preds.size(0)):
        [r, p] = pearsonr(np.squeeze(preds[i]), np.squeeze(targets[i]))
        rs.append(r)
    return rs

def mse_batch(preds, targets):
    mses = []
    for i in range(preds.size(0)):
        pred_tmp = np.array(np.squeeze(preds[i]))
        target_tmp = np.array(np.squeeze(targets[i]))
        mse = np.mean((pred_tmp - target_tmp) ** 2)
        mses.append(mse)
    return mses 

def cross_validate_nn(dataset, k, n_epochs, vowel_label, batch_size = 16, learning_rate = 1e-4):
    
    print("==== Running cross-validation for vowel %s ====" % (vowel_label))
    
    # Chech GPU
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        print("GPUs not available, use CPU")
    
    # Get the EEGs (X) and envelopes (Y) of the training data
    XTrain = dataset["XTrain"]
    YTrain = dataset["YTrain"]
    
    # Calculate fold size and indices
    fold_size = XTrain.shape[0] // k
    fold_indices = []
    for i in range(k):
        if i == k - 1:
            fold_indices.append((i * fold_size, XTrain.shape[0] - 1))
        else:
            fold_indices.append((i * fold_size, i * fold_size + fold_size - 1))
            
    # Set fixed random number seed
    torch.manual_seed(2021)
    
    # Initiate dictionary for storing the results
    results = {}
    results["cv_cor"] = {}
    results["cv_mse"] = {}
    results["targets"] = []
    results["preds"] = []
    
    # Iterate over k folds
    for fold_i in range(len(fold_indices)):
        
        # TO DO: Split the data into training and held-out (test) sets
        fold_start_i, fold_end_i = fold_indices[fold_i][0], fold_indices[fold_i][1]
        
        held_out_indices = np.arange(fold_start_i, fold_end_i + 1)
        XFold_train = XTrain[[x for x in range(XTrain.shape[0]) if x not in held_out_indices], :, :]
        YFold_train = YTrain[[x for x in range(YTrain.shape[0]) if x not in held_out_indices], :, :]
        XFold_out = XTrain[held_out_indices, :, :]
        YFold_out = YTrain[held_out_indices, :, :]
        print("    Cross-validating fold", fold_i + 1, "...")
    
        # Get data and labels for the training data and the held-out fold
        
        XFold_train = torch.Tensor(XFold_train)
        YFold_train = torch.Tensor(YFold_train)
        XFold_out = torch.Tensor(XFold_out)
        YFold_out = torch.Tensor(YFold_out)
        
        train_dataset = torch.utils.data.TensorDataset(XFold_train, YFold_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
        
        test_dataset = torch.utils.data.TensorDataset(XFold_out, YFold_out)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        
        # Initiate the model
        model = LSTM_basic()
        model = model.to(device)
        
        # Initialize optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        
        # Run for n_epochs
        for epoch in range(0, n_epochs):
            
            print("  ---- Starting epoch %d ----" % (epoch + 1), end = '\r')
            
            for i, (inputs, targets) in enumerate(train_loader, 0):
                
                # Send the inputs and targets to CPU or GPU
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                print("  Loss:%0.6f" % (loss.item()), end = '\r')
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                #scheduler.step()
                
        print("    Training process has finished for fold %d" % (fold_i + 1))
        print("    Saving the trained model")
        save_path = os.path.join(os.getcwd(), "models",  vowel_label + "_fold_" + str(fold_i + 1) + ".pth")
        torch.save(model.state_dict(), save_path)
        
        # Initiate lists to keep track of Pearson rs and Mean Squared Errors for the held-out set
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
                rs = pearsonr_batch(outputs, targets)
                mses = mse_batch(outputs, targets)
                
                fold_rs.extend(rs)
                fold_mses.extend(mses)
                results["targets"].extend([np.squeeze(x) for x in targets])
                results["preds"].extend([np.squeeze(x) for x in outputs])
                
            # Print results
            print("    Average Pearson r for fold %d: %0.6f" % (fold_i + 1, np.mean(fold_rs)))
            print("    Average MSE for fold %d: %0.8f" % (fold_i + 1, np.mean(fold_mses)))
            print('    --------------------------------')
        
        # Save the results
        results["cv_cor"][str(fold_i + 1)] = fold_rs
        results["cv_mse"][str(fold_i + 1)] = fold_mses
    
    return results