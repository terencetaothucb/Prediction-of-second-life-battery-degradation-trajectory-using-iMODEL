import argparse
import os
import torch
from exp.exp_forecasting import Exp_Long_Term_Forecast1
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='IMOE')

    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str,  default='IMOE',
                        help='IMOE,Informer,PATCHTST')

    parser.add_argument('--dataset', type=str, default='UL-NCA', help='')
    parser.add_argument('--condition', type=str, default='CY25-05_1', 
                        help='UL-NCA:CY45-05_1,CY25-05_1,CY25-025_1,CY25-1_1,CY35-05_1'
                        'UL-NCM:CY45-05_1,CY25-05_1,CY35-05_1,UL-NCMNCA:CY25-05_1,CY25-05_2,CY25-05_4,TPSL:Arbitrary,Fixed,LSD:LSD')
    parser.add_argument('--seq_len', type=int, default=50, help='')
    parser.add_argument('--enc_in', type=int, default=1, help='input sequence length')
    parser.add_argument('--hidden_dim', type=int, default=64, help='')
    parser.add_argument('--pred_len', type=int, default=50, help='prediction horizon')
    parser.add_argument('--num_experts', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--diverloss', type=int, default=0.5)
    parser.add_argument('--soc', type=int, default=20, help='20,30,40')
    parser.add_argument('--dataaccess', type=int, default=100, help='100,80,60,40')

    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--dropout', type=int, default=0.2, help='dimension of fcn')
    parser.add_argument('--patch_size', type=int, default=2, help='dimension of fcn')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=3000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--checkpoints', type=str, default='./checkpoint', help='location of model checkpoints')
    parser.add_argument('--patience', type=str, default=500, help='')
    parser.add_argument('--inverse', type=str, default='no', help='s')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')

    Exp = Exp_Long_Term_Forecast1

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args) 
            setting = '{}_ds{}_ex{}_pl{}_tk{}_dm{}'.format(
                args.model,               
                args.dataset,             
                args.num_experts,            
                args.pred_len,             
                args.top_k,            
                ii                        
            )   
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_ds{}_ex{}_pl{}_tk{}_dm{}'.format(
            args.model,              
            args.dataset,             
            args.num_experts,            
            args.pred_len,             
            args.top_k,          
            ii                       
        )   

        exp = Exp(args)  
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
