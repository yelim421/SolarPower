import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def apply_standard_scaling(train_file_path, test_file_path):
    # Load the datasets
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    
    # Drop the 'DateTime' column
    train_data_no_time = train_data.drop(columns=['DateTime'])
    test_data_no_time = test_data.drop(columns=['DateTime'])

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform
    train_scaled_array = scaler.fit_transform(train_data_no_time.values.astype('float32'))
    
    # Transform the test data
    test_scaled_array = scaler.transform(test_data_no_time.values.astype('float32'))
    
    
    # Convert the scaled arrays back to DataFrames
    train_scaled_df = pd.DataFrame(train_scaled_array, columns=train_data_no_time.columns)
    test_scaled_df = pd.DataFrame(test_scaled_array, columns=test_data_no_time.columns)
    print('------done apply_standard_scaling---------------')
    return train_scaled_df, test_scaled_df

def get_data_splits(train_scaled, test_scaled):
    train_X = train_scaled.drop(columns=['power'])
    test_X = test_scaled.drop(columns=['power'])
    train_y = train_scaled['power']
    test_y = test_scaled['power']
    #X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #((39205, 16), (39205,), (4470, 16), (4470,))
    print('------done get_data_splits---------------')
    return train_X, train_y, test_X, test_y


def get_possible_combinations():
    config = [[True, False], [16, 32, 64, 128], [8, 16, 32]]
    print('------done get_possible combinations---------------')
    return list(itertools.product(*config))

def create_and_train_model(train_X, train_y, test_X, test_y, n_neurons, n_batch_size, dropout, additional_layer):
    train_X2 = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    train_y2 = train_y.values.reshape((train_y.shape[0]))

    train_X3, valid_X, train_y3, valid_y = train_test_split(train_X2, train_y2, test_size = 0.2, shuffle = False)
    
    test_X2 = test_X.values.reshape((test_X.shape[0], 1,test_X.shape[1]))
    test_y2 = test_y.values#.reshape((test_y.shape[0]))

    model = Sequential()
    model.add(GRU(units=n_neurons, return_sequences=True, input_shape=(1, train_X.shape[1])))
    model.add(Dropout(dropout))
    if additional_layer:
        model.add(GRU(units=n_neurons, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(GRU(units=n_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(units=n_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(units=n_neurons, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='tanh'))

    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(f_name, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    
    model.fit(train_X3, train_y3, validation_data = (valid_X, valid_y), epochs=100, verbose=1, batch_size=n_batch_size, callbacks=[es, mc, rl], shuffle=False)
    print('------done create_and_train_model---------------')
    return model

def evaluate_hyperparameters(train_X, train_y):
    possible_combinations = get_possible_combinations()
    kfold = KFold(n_splits=5, shuffle=True)
    losses = np.empty(len(possible_combinations))
    
    for i, (additional_layer, n_neurons, n_batch_size) in enumerate(possible_combinations):
        print('--------------------------------------------------------------------')
        print(f'Combination #{i+1}: {additional_layer, n_neurons, n_batch_size}\n')
        
        val_loss = []
        for j, (train_index, val_index) in enumerate(kfold.split(train_X)):
            val_train_X = train_X.iloc[train_index, :]
            val_train_y = train_y.iloc[train_index]
            
            model = create_and_train_model(val_train_X, val_train_y, n_neurons, n_batch_size, 0.2, additional_layer)
            
            val_X = train_X.iloc[val_index, :].values.reshape((-1, 1, train_X.shape[1]))
            val_y = train_y.iloc[val_index].values
            
            val_accuracy = model.evaluate(val_X, val_y, verbose=0)
            val_loss.append(val_accuracy[1])
            
            print(f'{j+1}-FOLD ====> val RMSE: {val_accuracy[1]}')
        
        mean_val_loss = np.mean(val_loss)
        print(f'Mean validation RMSE: {mean_val_loss}')
        losses[i] = mean_val_loss
        
    best_index = np.argmin(losses)
    print(f"Best hyperparameters: {possible_combinations[best_index]} with validation RMSE: {losses[best_index]}")
    
    return possible_combinations[best_index], losses[best_index]

# Applying the functions

def main():
    train_scaled_df, test_scaled_df = apply_standard_scaling('train_data_datetime.csv', 'test_data_datetime.csv')
    train_X, train_y, test_X, test_y = get_data_splits(train_scaled_df, test_scaled_df)
    model = create_and_train_model(train_X, train_y, test_X, test_y, n_neurons, n_batch_size, dropout, additional_layer)
    train_X2 = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    train_y2 = train_y.values.reshape((train_y.shape[0]))
    test_X2 = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
    test_y2 = test_y.values
    #best_hyperparameters, best_loss = evaluate_hyperparameters(train_X, train_y)
    train_accuracy = model.evaluate(train_X2, train_y2, verbose=1)
    test_accuracy = model.evaluate(test_X2, test_y2, verbose=1)
    return train_accuracy, test_accuracy

# If this script is being run as the main module, execute the main function
if __name__ == "__main__":

    additional_layer = True
    n_neurons = 64
    n_batch_size = 32
    combination_no = 9
    dropout=0.2

    f_name = f'best_model{combination_no}.h5'  

    best_hyperparameters, best_loss = main()
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best validation RMSE: {best_loss}")