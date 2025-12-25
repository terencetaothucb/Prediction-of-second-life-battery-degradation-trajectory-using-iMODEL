import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader, Dataset
import random
import math

    
class BatteryDataset(Dataset):
    def __init__(self, data, window_size=50, capacity_length=50, scaler_features=None, soc=None):
        self.data = data
        self.window_size = window_size
        self.capacity_length = capacity_length
        self.soc = soc
        self.capacity_increment_list = []  
        self.features_list = []  

        for i in range(len(data)):
            relaxation_voltage = data.iloc[i]['Relaxation_Voltage']
            if isinstance(relaxation_voltage, str):  
                relaxation_voltage = eval(relaxation_voltage)  

            capacity_increment = data.iloc[i]['Capacity_Increment']
            if isinstance(capacity_increment, str):
                capacity_increment = eval(capacity_increment)

            capacity_increment = np.array(capacity_increment)

            if self.soc == 20:
                points_to_trim = 0  
            elif self.soc == 30:
                points_to_trim = 10  
            elif self.soc == 40:
                points_to_trim = 20 
            elif self.soc == 50:
                points_to_trim = 30 
            else:
                points_to_trim = 0   

            trimmed_ci = capacity_increment[points_to_trim:]  

            baseline = trimmed_ci[0]
            normalized_ci = trimmed_ci - baseline

            original_voltage_range = np.linspace(3.6, 4.15, 50)[points_to_trim:] 
            target_voltage_range = np.linspace(3.6, 4.15, 50)  

            capacity_increment = np.interp(
                target_voltage_range,
                original_voltage_range,
                normalized_ci
            )
            voltage_range = np.linspace(3.6, 4.15, len(capacity_increment))

            interpolated_voltage_range = np.linspace(3.6, 4.15, 1000)
            interpolated_capacity_increment = np.interp(
                interpolated_voltage_range,
                voltage_range,               
                capacity_increment        
            )


            idx_005 = np.argmin(np.abs(interpolated_voltage_range - 3.65))
            capacity_at_005 = interpolated_capacity_increment[idx_005]

            target_capacity = capacity_at_005 + 200
            target_idx = np.argmax(interpolated_capacity_increment >= target_capacity)
            voltage_at_target = interpolated_voltage_range[target_idx] if target_idx > 0 else np.nan

            relaxation_features = [
                np.mean(relaxation_voltage),  
                skew(relaxation_voltage),   
                np.max(relaxation_voltage),  
                np.min(relaxation_voltage),  
                np.var(relaxation_voltage),   
                kurtosis(relaxation_voltage, fisher=True) 
            ]

            capacity_increment_features = [
                np.mean(capacity_increment),  
                skew(capacity_increment),     
                np.max(capacity_increment),   
                np.var(capacity_increment),  
            ]

            features = np.concatenate([
                relaxation_features,         
                capacity_increment_features, 
                [capacity_at_005],          
                [voltage_at_target]        
            ])

            self.capacity_increment_list.append(capacity_increment)
            self.features_list.append(features)

        self.capacity_increment_list = np.array(self.capacity_increment_list)
        self.features_list = np.array(self.features_list)

        if scaler_features is None:
            self.features_list = self.features_list
        else:
            self.scaler_features = scaler_features
            self.features_list = self.scaler_features.transform(self.features_list)

    def __len__(self):
        return len(self.data) - self.window_size


    def __getitem__(self, idx):

        capacity_increment = self.capacity_increment_list[idx] / 1000  
        features = self.features_list[idx] 

        charge_current = self.data.iloc[idx:idx+self.window_size]['Charge_Current'].values 
        discharge_current = self.data.iloc[idx:idx+self.window_size]['Discharge_Current'].values
        Temperature = self.data.iloc[idx:idx+self.window_size]['Temperature'].values  

        capacity_increment = torch.tensor(capacity_increment, dtype=torch.float32)  
        features = torch.tensor(features, dtype=torch.float32)  
        charge_current = torch.tensor(charge_current, dtype=torch.float32)  
        discharge_current = torch.tensor(discharge_current, dtype=torch.float32)  
        Temperature = torch.tensor(Temperature, dtype=torch.float32)
        inputs = (capacity_increment, features, charge_current, discharge_current,Temperature)

        outputs = torch.tensor(
            self.data.iloc[idx:idx+self.window_size]['Discharge_Capacity'].values, dtype=torch.float32
        ) / 1000  

        return inputs, outputs
    
class BatteryDataset1(Dataset):
    def __init__(self, data, window_size=80, capacity_length=100, scaler_features=None,soc=None):
        self.data = data
        self.window_size = window_size
        self.capacity_length = capacity_length

        self.capacity_increment_list = [] 
        self.features_list = []  
        self.soc = soc
        for i in range(len(data)):
            capacity_increment = data.iloc[i]['QV_Curve']
            if isinstance(capacity_increment, str):  
                capacity_increment = eval(capacity_increment) 

            capacity_increment = np.array(capacity_increment)

            if self.soc == 20:
                points_to_trim = 0  
            elif self.soc == 30:
                points_to_trim = 10  
            elif self.soc == 40:
                points_to_trim = 20 
            elif self.soc == 50:
                points_to_trim = 30 
            else:
                points_to_trim = 0   

            trimmed_ci = capacity_increment[points_to_trim:]  

            baseline = trimmed_ci[0]
            normalized_ci = trimmed_ci - baseline

            original_voltage_range = np.linspace(3.6, 4.15, 50)[points_to_trim:]  
            target_voltage_range = np.linspace(3.6, 4.15, 50)  

            capacity_increment = np.interp(
                target_voltage_range,
                original_voltage_range,
                normalized_ci
            )
            voltage_range = np.linspace(3.6, 4.15, len(capacity_increment))

            interpolated_voltage_range = np.linspace(3.6, 4.15, 1000)
            interpolated_capacity_increment = np.interp(
                interpolated_voltage_range,
                voltage_range,               
                capacity_increment          
            )

            idx_005 = np.argmin(np.abs(interpolated_voltage_range - 3.65))
            capacity_at_005 = interpolated_capacity_increment[idx_005]

            target_capacity = capacity_at_005 + 0.2
            target_idx = np.argmax(interpolated_capacity_increment >= target_capacity)
            voltage_at_target = interpolated_voltage_range[target_idx] if target_idx > 0 else np.nan

            capacity_increment_features = [
                np.mean(capacity_increment), 
                skew(capacity_increment),     
                np.max(capacity_increment),  
                np.var(capacity_increment),  
                
            ]

            features = np.concatenate([
                capacity_increment_features,  
                [capacity_at_005],          
                [voltage_at_target]        
            ])

            self.capacity_increment_list.append(capacity_increment)
            self.features_list.append(features)

        self.capacity_increment_list = np.array(self.capacity_increment_list)
        self.features_list = np.array(self.features_list)

        if scaler_features is None:
            
            self.features_list = self.features_list
        else:

            self.scaler_features = scaler_features
            self.features_list = self.scaler_features.transform(self.features_list)

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        capacity_increment = self.capacity_increment_list[idx]   
        features = self.features_list[idx]  

        charge_current = self.data.iloc[idx:idx+self.window_size]['Charge_Current(A)'].values  
        discharge_current = self.data.iloc[idx:idx+self.window_size]['Discharge_Current(A)'].values
        Temperture = self.data.iloc[idx:idx+self.window_size]['Temperture'].values  
        capacity_increment = torch.tensor(capacity_increment, dtype=torch.float32)  
        features = torch.tensor(features, dtype=torch.float32)  
        charge_current = torch.tensor(charge_current, dtype=torch.float32)  
        discharge_current = torch.tensor(discharge_current, dtype=torch.float32)  
        Temperture = torch.tensor(Temperture, dtype=torch.float32)
        inputs = (capacity_increment, features, charge_current, discharge_current,Temperture)

        outputs = torch.tensor(
            self.data.iloc[idx:idx+self.window_size]['Max_Discharge_Capacity(Ah)'].values, dtype=torch.float32
        )  
        return inputs, outputs
    
class BatteryDataset2(Dataset):
    def __init__(self, data, window_size=50, capacity_length=50, scaler_features=None, soc=None):
        self.data = data
        self.window_size = window_size
        self.capacity_length = capacity_length
        self.soc = soc
        self.capacity_increment_list = []  
        self.features_list = []  

        for i in range(len(data)):
            relaxation_voltage = data.iloc[i]['Relaxation_Voltage']
            if isinstance(relaxation_voltage, str):  
                relaxation_voltage = eval(relaxation_voltage)
            capacity_increment = data.iloc[i]['Capacity_Increment']
            if isinstance(capacity_increment, str):
                capacity_increment = eval(capacity_increment)

            capacity_increment = np.array(capacity_increment)

            if self.soc == 20:
                points_to_trim = 0  
            elif self.soc == 30:
                points_to_trim = 10  
            elif self.soc == 40:
                points_to_trim = 20 
            elif self.soc == 50:
                points_to_trim = 30 
            else:
                points_to_trim = 0  

            trimmed_ci = capacity_increment[points_to_trim:]  

            baseline = trimmed_ci[0]
            normalized_ci = trimmed_ci - baseline

            original_voltage_range = np.linspace(3.6, 4.1, 50)[points_to_trim:]  
            target_voltage_range = np.linspace(3.6, 4.1, 50)  
            capacity_increment = np.interp(
                target_voltage_range,
                original_voltage_range,
                normalized_ci
            )
            voltage_range = np.linspace(3.6, 4.1, len(capacity_increment))

            interpolated_voltage_range = np.linspace(3.6, 4.1, 1000)
            interpolated_capacity_increment = np.interp(
                interpolated_voltage_range,  
                voltage_range,             
                capacity_increment      
            )
            idx_3_65 = np.argmin(np.abs(interpolated_voltage_range - 3.65))
            capacity_at_3_65 = interpolated_capacity_increment[idx_3_65]

            target_capacity = capacity_at_3_65 + 0.2
            target_idx = np.argmax(interpolated_capacity_increment >= target_capacity)
            voltage_at_target = interpolated_voltage_range[target_idx] if target_idx > 0 else np.nan

            relaxation_features = [
                np.mean(relaxation_voltage),  
                skew(relaxation_voltage),     
                np.max(relaxation_voltage),   
                np.min(relaxation_voltage),
                np.var(relaxation_voltage),  
                kurtosis(relaxation_voltage, fisher=True)  
            ]

            capacity_increment_features = [
                np.mean(capacity_increment),  
                np.max(capacity_increment),   
                np.var(capacity_increment),   
            ]

            features = np.concatenate([
                relaxation_features,         
                capacity_increment_features, 
                [capacity_at_3_65],         
                [voltage_at_target]      
            ])

            self.capacity_increment_list.append(capacity_increment)
            self.features_list.append(features)

        self.capacity_increment_list = np.array(self.capacity_increment_list)
        self.features_list = np.array(self.features_list)

        if scaler_features is None:
            self.features_list = self.features_list
        else:
            self.scaler_features = scaler_features
            self.features_list = self.scaler_features.transform(self.features_list)

    def __len__(self):
        return len(self.data) - self.window_size


    def __getitem__(self, idx):

        capacity_increment = self.capacity_increment_list[idx]  
        features = self.features_list[idx]  

        charge_current = self.data.iloc[idx:idx+self.window_size]['Charge_Current'].values 
        discharge_current = self.data.iloc[idx:idx+self.window_size]['Discharge_Current'].values
        Temperature = self.data.iloc[idx:idx+self.window_size]['Temperature'].values
        capacity_increment = torch.tensor(capacity_increment, dtype=torch.float32)  
        features = torch.tensor(features, dtype=torch.float32) 
        charge_current = torch.tensor(charge_current, dtype=torch.float32)  
        discharge_current = torch.tensor(discharge_current, dtype=torch.float32)  
        Temperature = torch.tensor(Temperature, dtype=torch.float32)
        inputs = (capacity_increment, features, charge_current, discharge_current,Temperature)
        outputs = torch.tensor(
            self.data.iloc[idx:idx+self.window_size]['Discharge_Capacity'].values, dtype=torch.float32
        )  
        return inputs, outputs


def NCA_trainloader(args):
    train_samples = []  
    train_features_list = []  
    train_outputs = []  
    val_samples = []  
    val_outputs = []  

    test_samples = [] 
    test_outputs = []  

    if args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv',
            'CY45-05_1-#17.csv'
        ]
        val_files = [
            'CY45-05_1-#28.csv', 'CY45-05_1-#25.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]

    elif args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv',
            'CY25-05_1-#5.csv', 'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#13.csv'
        ]
        val_files = [
            'CY25-05_1-#18.csv', 'CY25-05_1-#19.csv'
        ]
        test_files = [
            'CY25-05_1-#1.csv', 'CY25-05_1-#14.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv',
            'CY25-05_1-#17.csv', 'CY25-05_1-#12.csv'
        ]

    elif args.condition == 'CY25-025_1':
        train_files = [
            'CY25-025_1-#1.csv', 'CY25-025_1-#2.csv', 'CY25-025_1-#3.csv'
        ]
        val_files = [
            'CY25-025_1-#7.csv'
        ]
        test_files = [
            'CY25-025_1-#5.csv', 'CY25-025_1-#6.csv', 'CY25-025_1-#4.csv'
        ]

    elif args.condition == 'CY25-1_1':
        train_files = [
            'CY25-1_1-#1.csv', 'CY25-1_1-#2.csv', 'CY25-1_1-#3.csv', 'CY25-1_1-#4.csv', 'CY25-1_1-#5.csv'
        ]
        val_files = [
            'CY25-1_1-#6.csv'
        ]
        test_files = [
            'CY25-1_1-#7.csv', 'CY25-1_1-#8.csv', 'CY25-1_1-#9.csv'
        ]

    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv'
        ]

    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    
    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:  
            train_files = train_files.copy()
        else:

            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
        
            train_files = random.sample(train_files, num_train)
    else:  
        train_files = train_files.copy()
    input_folder = 'dataset/UL-NCA/'

    for file_name in train_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx] 
            train_samples.append(inputs)  
            train_features_list.append(inputs[1].numpy())  
            train_outputs.append(outputs) 

    train_features_list = np.array(train_features_list)
    scaler_features = StandardScaler()
    train_features_list = scaler_features.fit_transform(train_features_list)

    for i in range(len(train_samples)):
        original_sample = train_samples[i]
        updated_sample = (
            original_sample[0], 
            torch.tensor(train_features_list[i], dtype=torch.float32), 
            original_sample[2],  
            original_sample[3],
            original_sample[4]   
        )

        train_samples[i] = updated_sample

    for file_name in val_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx] 
            val_samples.append(inputs)  
            val_outputs.append(outputs)  

    for file_name in test_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]  
            test_samples.append(inputs)  
            test_outputs.append(outputs)  

    train_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(train_samples, train_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(val_samples, val_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(test_samples, test_outputs)],
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler_features

def NCM_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []

    if args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv'
        ]
        val_files = [
            'CY45-05_1-#28.csv','CY45-05_1-#17.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]
    elif args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv', 'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv',
            'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#12.csv',
            'CY25-05_1-#13.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv'
        ]
        val_files = [
            'CY25-05_1-#5.csv','CY25-05_1-#17.csv'
        ]
        test_files = [
            'CY25-05_1-#18.csv', 'CY25-05_1-#19.csv', 'CY25-05_1-#20.csv', 'CY25-05_1-#21.csv',
            'CY25-05_1-#22.csv', 'CY25-05_1-#23.csv'
        ]
    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv', 'CY35-05_1-#4.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    input_folder = 'dataset/UL-NCM/'

    for file_name in train_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))
        battery_dataset = BatteryDataset(data, window_size=args.pred_len,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]
            train_samples.append(inputs)
            train_features_list.append(inputs[1].numpy())
            train_outputs.append(outputs)

    train_features_list = np.array(train_features_list)
    scaler_features = StandardScaler()
    train_features_list = scaler_features.fit_transform(train_features_list)

    for i in range(len(train_samples)):
        original_sample = train_samples[i]
        updated_sample = (
            original_sample[0],
            torch.tensor(train_features_list[i], dtype=torch.float32),
            original_sample[2],
            original_sample[3],
            original_sample[4]
        )
        train_samples[i] = updated_sample

    for file_name in val_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))
        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]
            val_samples.append(inputs)
            val_outputs.append(outputs)

    for file_name in test_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))
        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]
            test_samples.append(inputs)
            test_outputs.append(outputs)

    train_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(train_samples, train_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(val_samples, val_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(test_samples, test_outputs)],
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler_features

def NCMNCA_trainloader(args):
    train_samples = []  
    train_features_list = []  
    train_outputs = [] 
    val_samples = [] 
    val_outputs = [] 
    test_samples = [] 
    test_outputs = []  

    if args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv'
        ]
        val_files = [
            'CY25-05_1-#2.csv'
        ]
        test_files = [
            'CY25-05_1-#3.csv'
        ]

    elif args.condition == 'CY25-05_2':
        train_files = [
            'CY25-05_2-#1.csv'
        ]
        val_files = [
            'CY25-05_2-#2.csv'
        ]
        test_files = [
            'CY25-05_2-#3.csv'
        ]

    elif args.condition == 'CY25-05_4':
        train_files = [
            'CY25-05_4-#1.csv'
        ]
        val_files = [
            'CY25-05_4-#2.csv'
        ]
        test_files = [
            'CY25-05_4-#3.csv'
        ]

    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:  
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))

            train_files = random.sample(train_files, num_train)
    else: 
        train_files = train_files.copy()
    input_folder = 'dataset/UL-NCMNCA/'

    for file_name in train_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]
            train_samples.append(inputs) 
            train_features_list.append(inputs[1].numpy()) 
            train_outputs.append(outputs)  

    train_features_list = np.array(train_features_list)
    scaler_features = StandardScaler()
    train_features_list = scaler_features.fit_transform(train_features_list)

    for i in range(len(train_samples)):

        original_sample = train_samples[i]
        updated_sample = (
            original_sample[0],  
            torch.tensor(train_features_list[i], dtype=torch.float32), 
            original_sample[2],
            original_sample[3],
            original_sample[4]   
        )

        train_samples[i] = updated_sample

    for file_name in val_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx] 
            val_samples.append(inputs) 
            val_outputs.append(outputs) 

    for file_name in test_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx]  
            test_samples.append(inputs) 
            test_outputs.append(outputs)  

    train_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(train_samples, train_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(val_samples, val_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(test_samples, test_outputs)],
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler_features

def TPSL_trainloader(args):

    train_samples = []  
    train_features_list = []  
    train_outputs = []  

    val_samples = []  
    val_outputs = []  

    test_samples = []  
    test_outputs = []  
    if args.condition == 'Arbitrary':
        train_file_paths = [
            '#1', '#3', '#7', '#9', '#14', '#15', '#17', '#18', '#20', '#21', '#24', '#25', '#27', '#28',
            '#30', '#31',  '#34', '#36', '#37', '#39', '#40', '#42',  '#46', '#47', '#50', '#54', '#55', '#56', '#59', '#60', 
            '#74', '#75', '#76', '#77','#67', '#68', '#69','#73'
        ]
        val_file_paths = [
            '#66','#70'
        ]
        test_file_paths = [
            '#5','#8', '#11', '#12', '#71', '#72', '#33','#43', '#61', '#62', '#63', '#64', '#65'
        ]
        input_folder = 'dataset/TPSL-Arbitrary'
        
    elif args.condition == 'Fixed':
        train_file_paths = [
            '#6', '#22',  '#26', '#29', '#32',  '#38', '#41', '#44',   '#49', '#52',
            '#53'
 
        ]
        val_file_paths = [
            '#45', '#58'
        ]

        test_file_paths = [
            '#23',  '#35', '#48', '#57'
        ]

        input_folder = 'dataset/TPSL-Fixed'

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:  
            train_file_paths = train_file_paths.copy()
        else:
            num_train = max(1, math.ceil(len(train_file_paths) * args.dataaccess / 100))
            
            train_file_paths = random.sample(train_file_paths, num_train)
    else:  
        train_file_paths = train_file_paths.copy()
    for folder_name in train_file_paths:
        file_path = os.path.join(input_folder, folder_name, 'combined_data.csv')
        data = pd.read_csv(file_path)
        data['QV_Curve'] = data['QV_Curve'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset1(data, window_size=args.pred_len,soc=args.soc)
        for idx in range(19):
            inputs, outputs = battery_dataset[idx]  
            train_samples.append(inputs)  
            train_features_list.append(inputs[1].numpy())  
            train_outputs.append(outputs)  

    train_features_list = np.array(train_features_list)
    scaler_features = StandardScaler()
    train_features_list = scaler_features.fit_transform(train_features_list)

    for i in range(len(train_samples)):
        original_sample = train_samples[i]
        updated_sample = (
            original_sample[0],  
            torch.tensor(train_features_list[i], dtype=torch.float32), 
            original_sample[2],  
            original_sample[3], 
            original_sample[4] 
        )
        train_samples[i] = updated_sample

    for folder_name in val_file_paths:
        file_path = os.path.join(input_folder, folder_name, 'combined_data.csv')
        data = pd.read_csv(file_path)
        data['QV_Curve'] = data['QV_Curve'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset1(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(19):
            inputs, outputs = battery_dataset[idx]  
            val_samples.append(inputs)  
            val_outputs.append(outputs) 

    for folder_name in test_file_paths:
        file_path = os.path.join(input_folder, folder_name, 'combined_data.csv')
        data = pd.read_csv(file_path)
        data['QV_Curve'] = data['QV_Curve'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset1(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(19):
            inputs, outputs = battery_dataset[idx]  
            test_samples.append(inputs)  
            test_outputs.append(outputs) 


    train_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(train_samples, train_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(val_samples, val_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(test_samples, test_outputs)],
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler_features

def LSD_trainloader(args):
    train_samples = []  
    train_features_list = []  
    train_outputs = []  
    val_samples = []  
    val_outputs = []  
    test_samples = [] 
    test_outputs = []  

    if args.condition == 'LSD':

        train_files = [

        '1.csv', '10.csv', '11.csv', '12.csv', '13.csv', '14.csv', '15.csv', '16.csv', '17.csv', '18.csv',

        '19.csv', '2.csv', '20.csv', '21.csv', '22.csv', '23.csv', 

        '28.csv', '29.csv', '3.csv', '30.csv', '31.csv', '32.csv', '33.csv', '34.csv', '35.csv', '36.csv',

        '37.csv', '38.csv', '39.csv', '4.csv', '40.csv', '41.csv', '42.csv',  

        '46.csv', '47.csv', '48.csv', '49.csv', '5.csv', '50.csv', '51.csv', '52.csv', '53.csv', '54.csv',

        '55.csv', '56.csv', '57.csv', '58.csv', '59.csv', '6.csv', '60.csv', '63.csv', '64.csv', '65.csv',

        '80.csv', '81.csv', '82.csv', '83.csv',

        ]
        val_files = [

        '66.csv', '67.csv', '68.csv', '69.csv', '7.csv', '70.csv', '71.csv', '72.csv', '73.csv', '74.csv','86.csv', '87.csv',

        ]
        test_files = [

        '43.csv', '44.csv', '45.csv','75.csv', '76.csv', '77.csv', '78.csv', '79.csv', '8.csv', '24.csv', '25.csv', '26.csv', '27.csv',

        '84.csv', '85.csv',  '88.csv', '9.csv'

        ]

    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    
    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:  
            train_files = train_files.copy()
        else:

            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))

            train_files = random.sample(train_files, num_train)
    else:  
        train_files = train_files.copy()
    input_folder ='dataset/LSD'

    # 加载训练数据
    for file_name in train_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset(data, window_size=args.pred_len,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx+1]  
            train_samples.append(inputs)  
            train_features_list.append(inputs[1].numpy())  
            train_outputs.append(outputs)  

    train_features_list = np.array(train_features_list)
    scaler_features = StandardScaler()
    train_features_list = scaler_features.fit_transform(train_features_list)

    for i in range(len(train_samples)):
        original_sample = train_samples[i]
        updated_sample = (
            original_sample[0],  
            torch.tensor(train_features_list[i], dtype=torch.float32),  
            original_sample[2],  
            original_sample[3],
            original_sample[4]    
        )
        train_samples[i] = updated_sample
    for file_name in val_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset2(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx+1] 
            val_samples.append(inputs)  
            val_outputs.append(outputs) 
    for file_name in test_files:
        file_path = os.path.join(input_folder, file_name)
        data = pd.read_csv(file_path)
        data['Capacity_Increment'] = data['Capacity_Increment'].apply(lambda x: eval(x))
        data['Relaxation_Voltage'] = data['Relaxation_Voltage'].apply(lambda x: eval(x))

        battery_dataset = BatteryDataset2(data, window_size=args.pred_len, scaler_features=scaler_features,soc=args.soc)
        for idx in range(len(battery_dataset)):
            inputs, outputs = battery_dataset[idx+1]  
            test_samples.append(inputs)  
            test_outputs.append(outputs)  

    train_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(train_samples, train_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(val_samples, val_outputs)],
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        [(inputs, outputs) for inputs, outputs in zip(test_samples, test_outputs)],
        batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler_features