# from data_loader import *
from exp.exp_basic import Exp_Basic
from dataloader import *
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate, visual
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
import time
def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)

class Exp_Long_Term_Forecast1(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast1, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self):
        if self.args.dataset == 'UL-NCA':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCA_trainloader(self.args)
        elif self.args.dataset == 'UL-NCM':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCM_trainloader(self.args)
        elif self.args.dataset == 'UL-NCMNCA':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCMNCA_trainloader(self.args)
        elif self.args.dataset == 'TPSL':
            train_loader, X_test_tensor, y_test_tensor, scaler = TPSL_trainloader(self.args)
        elif self.args.dataset == 'LSD':
            train_loader, X_test_tensor, y_test_tensor, scaler = LSD_trainloader(self.args)
        return train_loader, X_test_tensor, y_test_tensor,scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_loader, val_loader, test_loader, scaler = self._get_data()

        time_now = time.time()
        path = os.path.join(self.args.checkpoints,self.args.model, self.args.dataset ,self.args.condition,setting)
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_diversity_loss= []
            

            self.model.train()
            epoch_time = time.time()
            for inputs, targets in train_loader:
                capacity_increment, relaxation_features, charge_current, discharge_current,Temperature = inputs
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = targets.to(self.device)

                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current,Temperature)

                if self.args.model == 'IMOE':
                    importance = gates.sum(0)  
                    diversity_loss = cv_squared(importance) 
                else:
                    diversity_loss = torch.tensor(0.0, device=self.device) 

                main_loss = criterion(outputs, targets)

                total_loss = main_loss + self.args.diverloss * diversity_loss
                train_loss.append(total_loss.item())
                train_diversity_loss.append(diversity_loss.item())  

                model_optim.zero_grad()
                total_loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            train_diversity_loss = np.average(train_diversity_loss)
            vali_loss = self.vali(train_loader, val_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} diversity Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, train_diversity_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
            

        return self.model
    
    def vali(self, train_loader, val_loader, criterion):
        self.model.eval()  
        
        val_loss = [] 
        
        with torch.no_grad():  
            for i, (batch_x, batch_y) in enumerate(val_loader):
                capacity_increment, relaxation_features, charge_current, discharge_current,Temperature = batch_x
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = batch_y.to(self.device)
 
                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current,Temperature)

                if self.args.model == 'IMOE':
                    importance = gates.sum(0)  
                    diversity_loss = cv_squared(importance)  
                else:
                    diversity_loss = 0  #
 
                main_loss = criterion(outputs, targets)
                
                total_loss = main_loss + self.args.diverloss * diversity_loss
                val_loss.append(total_loss.item())  

        avg_val_loss = np.mean(val_loss)
        self.model.train()
        return avg_val_loss


    def test(self, setting, test=0):
        train_loader, val_loader, test_loader, scaler = self._get_data()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('checkpoint\IMOE\UL-NCA\CY45-05_1\IMOE_dsNCA_ex5_pl50_tk2_ep2000_dm0\checkpoint.pth')))
        self.model.eval()

        data_loader = test_loader
        total_rmse = 0
        total_mape = 0
        count = 0
        all_true_values = []
        all_pred_values = []
        all_weights = []
        start_time = time.time()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                capacity_increment, relaxation_features, charge_current, discharge_current, Temperature = X_batch
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = y_batch.to(self.device)

                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current, Temperature)

                all_weights.append(gates.cpu().numpy())

                if self.args.inverse == 'yes':  
                    y_pred_inv = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).reshape(outputs.shape)
                    y_batch_inv = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).reshape(targets.shape)
                else:
                    y_pred_inv = outputs.cpu().numpy().reshape(outputs.shape)
                    y_batch_inv = targets.cpu().numpy().reshape(targets.shape)

                eps = 1e-8  
                diff = y_pred_inv - y_batch_inv

                rmse = np.sqrt(np.nanmean(diff ** 2))
                denom = np.where(np.abs(y_batch_inv) < eps, np.nan, np.abs(y_batch_inv))
                mape = np.nanmean(np.abs(diff / denom)) * 100.0
                total_rmse += rmse
                total_mape += mape
                count += 1
                all_true_values.append(y_batch_inv)
                all_pred_values.append(y_pred_inv)
        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"Total test time: {total_test_time:.2f} seconds")
        avg_rmse = total_rmse / count
        avg_mape = total_mape / count
        print(f"Average MAPE (Normalized): {avg_mape:.4f}%")
        print(f"Average Test RMSE: {avg_rmse:.4f}")
        all_true_values = np.concatenate(all_true_values, axis=0)
        all_pred_values = np.concatenate(all_pred_values, axis=0)
        all_weights = np.concatenate(all_weights, axis=0)
        output_dir = os.path.join(self.args.checkpoints, self.args.model, self.args.dataset, self.args.condition,setting)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'true_values.npy'), all_true_values)
        np.save(os.path.join(output_dir, 'pred_values.npy'), all_pred_values)
        if self.args.model == 'IMOE':
            weights_df = pd.DataFrame(all_weights)
            weights_df.to_csv(os.path.join(output_dir, 'weights.csv'), index=False)
        colors = ["#403990", "#80A6E2", "#FBBD85", "#F46F43", "#CF3D3E"]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
        if self.args.model == 'IMOE':
            plt.figure(figsize=(12, 6))
            sns.heatmap(all_weights.T, cmap=cmap, cbar=True, yticklabels=['Weight 1', 'Weight 2', 'Weight 3'])
            plt.title('Weights for Each Sample')
            plt.xlabel('Sample Index')
            plt.ylabel('Weights')
            plt.savefig(os.path.join(output_dir, 'weights_heatmap.png')) 
            plt.close()  
        plt.figure(figsize=(15, 8))
        for i in range(len(all_true_values)):
            plt.plot(range(i, i + len(all_true_values[i])), all_true_values[i], color='blue', alpha=0.5, label='True Values' if i == 0 else "")
        for i in range(len(all_pred_values)):
            plt.plot(range(i, i + len(all_pred_values[i])), all_pred_values[i], color='red', alpha=0.5, label='Predictions' if i == 0 else "")
        plt.legend()
        plt.title('All Samples: True Values vs Predictions')
        plt.xlabel('Cycle Index')
        plt.ylabel('Discharge Capacity')
        plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png')) 
        plt.close()  
        return avg_rmse, avg_mape