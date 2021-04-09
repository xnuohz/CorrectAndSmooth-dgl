import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        self.linears.append(nn.Linear(hid_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = bn(linear(x))
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return F.log_softmax(x, dim=-1)


class LabelPropagation(nn.Module):
    r"""
    
    """
    def __init__(self, num_layers, alpha):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
    
    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            
            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y

            for _ in range(self.num_layers):
                degs = g.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(1)
                g.ndata['h'] = y * norm
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = last + self.alpha * g.ndata.pop('h') * norm
                y = post_step(y)
            
            return y


class CorrectAndSmooth(nn.Module):
    r"""
    
    """
    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 num_smoothing_layers,
                 smoothing_alpha,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()
        
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def correct(self, g, y_soft, y_true, mask):
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0
                return y_soft + scale * smoothed_error
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x
                
                smoothed_error = self.prop1(g, error, post_step=fix_input)
                return y_soft + self.scale * smoothed_error

    def smooth(self, g, y_soft, y_true, mask):
        with g.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            y_soft[mask] = y_true
            return self.prop2(g, y_soft)
