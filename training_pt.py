import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import tensorflow as tf
from sklearn.model_selection import KFold
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    
    return train_scaled_df, test_scaled_df

def get_data_splits(train_scaled, test_scaled):
    train_X = train_scaled.drop(columns=['power'])
    test_X = test_scaled.drop(columns=['power'])
    train_y = train_scaled['power']
    test_y = test_scaled['power']
    #X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #((39205, 16), (39205,), (4470, 16), (4470,))
    return train_X, train_y, test_X, test_y


def get_possible_combinations():
    config = [[True, False], [16, 32, 64, 128], [8, 16, 32]]
    return list(itertools.product(*config))



def create_and_train_model(train_X, train_y, n_neurons, n_batch_size, dropout, additional_layer):
    val_train_X2 = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_train_y2 = train_y.values.reshape((train_y.shape[0]))
    
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
    
    model.fit(val_train_X2, val_train_y2, epochs=100, verbose=0, batch_size=n_batch_size, callbacks=[es], shuffle=False)
    
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

# def main():
#     train_scaled_df, test_scaled_df = apply_standard_scaling('train_data_datetime.csv', 'test_data_datetime.csv')
#     train_X, train_y, test_X, test_y = get_data_splits(train_scaled_df, test_scaled_df)
#     best_hyperparameters, best_loss = evaluate_hyperparameters(train_X, train_y)
#     return best_hyperparameters, best_loss

# # If this script is being run as the main module, execute the main function
# if __name__ == "__main__":
#     best_hyperparameters, best_loss = main()
#     print(f"Best hyperparameters: {best_hyperparameters}")
#     print(f"Best validation RMSE: {best_loss}")

class GRUModel(nn.Module):
    def __init__(self, input_dim, n_neurons, dropout, additional_layer):
        super(GRUModel, self).__init__()
        
        self.additional_layer = additional_layer
        
        self.gru1 = nn.GRU(input_dim, n_neurons, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        if self.additional_layer:
            self.gru2 = nn.GRU(n_neurons, n_neurons, batch_first=True)
            self.dropout2 = nn.Dropout(dropout)
            
        self.gru3 = nn.GRU(n_neurons, n_neurons, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.gru4 = nn.GRU(n_neurons, n_neurons, batch_first=True)
        self.dropout4 = nn.Dropout(dropout)
        
        self.gru5 = nn.GRU(n_neurons, n_neurons, batch_first=True)
        self.dropout5 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(n_neurons, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        
        if self.additional_layer:
            x, _ = self.gru2(x)
            x = self.dropout2(x)
            
        x, _ = self.gru3(x)
        x = self.dropout3(x)
        
        x, _ = self.gru4(x)
        x = self.dropout4(x)
        
        x, _ = self.gru5(x)
        x = self.dropout5(x)
        
        x = self.fc(x[:, -1, :])
        x = self.tanh(x)
        
        return x
    
def main():
    # DataLoader 정의
    train_X, train_y, _, _ = get_data_splits(*apply_standard_scaling('train_data_datetime.csv', 'test_data_datetime.csv'))
    train_dataset = TensorDataset(torch.tensor(train_X.values, dtype=torch.float32), torch.tensor(train_y.values, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=n_batch_size, shuffle=True)

    # 모델, 최적화기, 손실 함수 정의
    model = GRUModel(input_dim=train_X.shape[1], n_neurons=n_neurons, dropout=0.2, additional_layer=additional_layer)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # 학습
    for epoch in range(100):  # epoch 수는 임의로 설정
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.tensor(train_X.values, dtype=torch.float32))

if __name__ == "__main__":
    main()