import numpy as np
from b2sd_nn_model_helpers import preprocess_data, cross_validate_nn

if __name__ == '__main__':
    
    # Configurations
    n_epochs = 50
    k = 10
    print("Run neural network models using the following configurations:")
    print("No. epochs:", n_epochs)
    print("No. folds (k):", k)
    
    # Load the training and test data
    print("\nLoading the data...")
    train_dat = np.load("data_top5v/data_train_top5v.npz")
    test_dat = np.load("data_top5v/data_test_top5v.npz")
    
    # Preprocessing the data
    print("Preprocessing the data...")
    datasets_by_vowel = preprocess_data(train_dat, test_dat)
    # loggin nodde: srun -c 1 -n -1 -- pty bash
    
    # For each vowel category, run LSTM models and perform K-fold cross-validation
    print("\nNow start training")
    print("Training LSTM models for each vowel category and perform k-fold cross-validation")
    lstm_results = {}
    for v in datasets_by_vowel.keys():
        lstm_results[v] = cross_validate_nn(datasets_by_vowel[v], k, n_epochs, v)
    print("Done!")
    
    # Save the results
    np.savez("models/lstm_results.npz", lstm_results = lstm_results)
    print("Results saved!")