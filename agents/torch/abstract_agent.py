from typing import Dict, Tuple, Union
import torch
import os


class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda")
        self.curr_nets = {}
        self.targ_nets = {}
        self.num_update_calls = 0
        
    def get_action(self, 
                   obs: Union[Dict[str, torch.Tensor], torch.Tensor],
                   eval_mode: bool):
        # Input tensors have shape (1, *)
        pass
    
    def update(self, 
               batch: Tuple[Union[Dict[str, torch.Tensor], torch.Tensor], 
                            torch.Tensor,
                            torch.Tensor,
                            Union[Dict[str, torch.Tensor], torch.Tensor],
                            torch.Tensor]):
        # Input tensors have shape (batch_size, *)
        pass
    
    def _set_network_states(self, state):
        # Set network state to train or eval for batchnorm and dropout
        assert state in ["eval", "train"]
        
        for _, net in self.curr_nets.items():
            if state == "eval":
                net.eval()
            else:
                net.train()
            
    def _put_on_device(self, x):
        # Put tensor or dictionary of tensors on GPU
        if isinstance(x, dict):
            x = {k: v.to(self.device) for k, v in x.items()}
        else:
            x = x.to(self.device)
        return x
    
    def _soft_update_target(self):
        # Update frozen target models
        for name, net in self.curr_nets.items():
            targ_net = self.targ_nets[name]
            
            for targ_param, param in zip(targ_net.parameters(), net.parameters()):
                targ_param.data.copy_(targ_param.data * (1.0 - self.target_lr) + \
                                      param.data * self.target_lr)
                
    def _log_weights_and_grads(self, directory, keys, weights, grads):
        # For every network in keys log weights/gradients of every parameter
        for net_name in keys:
            if net_name in self.curr_nets:
                for p_name, p in self.curr_nets[net_name].named_parameters():
                    path = os.path.join(directory, net_name, p_name)
                    weights[path] = p.clone().cpu().data.numpy().flatten()
                    grads[path] = p.grad.clone().cpu().data.numpy().flatten()
            
    def save(self, filename):
        for name, net in self.curr_nets.items():
            torch.save(net.state_dict(), filename + name + ".pth")
        
    def load(self, filename):
        for name, net in self.curr_nets.items():
            try:
                net.load_state_dict(torch.load(filename + name + ".pth"))
            except:
                print("Could not load " + name)
                