import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types
from torch import nn
import torch

from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device

def _layernorm(hidden_states, eps=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)

def _monkeypatch_mlp(mlp, file_path, grabbing_mode=False):
    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward, mlp)

    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode
    mlp.norm2 = nn.LayerNorm(mlp.intermediate_size, elementwise_affine=False)

    if not grabbing_mode:
        mlp.distrs = {}
        mlp.distrs['gate'] = Distribution(file_path, hidden_type='gate')
        mlp.distrs['up'] = Distribution(file_path, hidden_type='up')
        mlp.distrs['down'] = Distribution(file_path, hidden_type='down')


        mlp.sparse_fns = nn.ModuleDict({
            'gate': SparsifyFn(mlp.distrs['gate']).to(get_module_device(mlp)),
            'up': SparsifyFn(mlp.distrs['up']).to(get_module_device(mlp)),
            'down': SparsifyFn(mlp.distrs['down']).to(get_module_device(mlp)),
        })


    mlp.activation_module = ActivationModule(file_path)

    return mlp

def _mlp_forward(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            
            gate_tmp = x * torch.norm(self.gate_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            up_tmp = x * torch.norm(self.up_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            self.activation_module.grab_activations(gate_tmp, 'gate')
            self.activation_module.grab_activations(up_tmp, 'up')
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            grab_temp = intermediate_states * torch.norm(self.down_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            self.activation_module.grab_activations(grab_temp, 'down')
            down_proj = self.down_proj(intermediate_states)
                
        else:

            up_norm = torch.norm(self.up_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            down_norm = torch.norm(self.down_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            gate_norm = torch.norm(self.gate_proj.weight.data, dim=0, p=1, dtype=torch.float32)
            x_gate = (self.sparse_fns['gate'](x * gate_norm) / (gate_norm + 1e-4)).to(torch.float16)
            x_up = (self.sparse_fns['up'](x * up_norm) / (up_norm + 1e-4)).to(torch.float16)
            intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
            intermediate_states = (self.sparse_fns['down'](intermediate_states * down_norm) / (down_norm + 1e-4)).to(torch.float16)
            down_proj = self.down_proj(intermediate_states)

    return down_proj