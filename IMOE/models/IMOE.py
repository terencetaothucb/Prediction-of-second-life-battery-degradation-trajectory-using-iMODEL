import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_experts = args.num_experts 
        self.capacity_fc = nn.ModuleList([nn.Linear(args.seq_len, args.pred_len) for _ in range(self.num_experts)])
        input_dim = 6 if args.dataset == 'TPSL' else 12
        self.features_fc = nn.Linear(input_dim, self.num_experts)
        self.softmax = nn.Softmax(dim=1)
        self.top_k = args.top_k
        self.alpha = args.alpha
        self.w_noise = nn.Parameter(torch.zeros(self.num_experts, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.noise_epsilon = 1e-5
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=4, hidden_size=args.hidden_dim, batch_first=True, bidirectional=True) 
        self.fc_final = nn.Linear(2 * args.hidden_dim, 1)

    def decompostion_tp(self, x):
        output = torch.zeros_like(x)
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = self.alpha * torch.log(x[mask] + 1)
        output[~mask] = self.alpha * (torch.exp(x[~mask]) - 1)
        return output

    def forward(self, capacity_increment, features, charge_current, discharge_current, Temperature):
        outs = [fc(capacity_increment) for fc in self.capacity_fc]
        raw_weights = self.features_fc(features)
        clean_logits = raw_weights
        if self.training:
            raw_noise_stddev = clean_logits @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        weights = self.decompostion_tp(logits) 
        weights = self.softmax(weights)
        Degradation_trend = torch.zeros_like(outs[0])
        for i in range(self.num_experts):
            Degradation_trend += weights[:, i].unsqueeze(1) * outs[i]
        charge_current = charge_current.unsqueeze(2) 
        discharge_current = discharge_current.unsqueeze(2) 
        Temperature = Temperature.unsqueeze(2) 
        lstm_input = torch.cat([Degradation_trend.unsqueeze(2), charge_current, discharge_current, Temperature], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        final_output = self.fc_final(lstm_out)

        return final_output.squeeze(-1), weights