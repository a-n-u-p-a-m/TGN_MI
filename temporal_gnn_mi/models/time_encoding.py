import torch
import torch.nn as nn
from torch import Tensor

class TimeEncoding(nn.Module):
    """Learnable temporal encoding using scaled sinusoidal features.
    Phi(dt) = [cos(w1*dt), sin(w1*dt), ..., cos(wd*dt), sin(wd*dt)]
    where w_i are learnable parameters.
    
    HOOK POINT: self.hook_data
      Captures the raw time encodings for every forward call.
    """
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Initialize learnable frequency parameters
        self.w = nn.Parameter(torch.empty(1, time_dim // 2))
        nn.init.xavier_uniform_(self.w)
        self.hook_data = []

    def forward(self, delta_t: Tensor) -> Tensor:
        """
        Args:
            delta_t: Tensor of shape [batch_size] containing time differences
        Returns:
            Tensor of shape [batch_size, time_dim]
        """
        # [batch_size, 1] * [1, time_dim // 2] -> [batch_size, time_dim // 2]
        scaled_time = delta_t.view(-1, 1) * self.w
        
        # [batch_size, time_dim]
        encoding = torch.cat([torch.cos(scaled_time), torch.sin(scaled_time)], dim=-1)
        
        # If time_dim is odd, pad with zeros
        if self.time_dim % 2 != 0:
            encoding = torch.cat([encoding, torch.zeros(encoding.size(0), 1, device=encoding.device)], dim=-1)
        
        # Hook: capture the time encoding and raw delta_t
        self.hook_data.append({
            'delta_t': delta_t.clone().detach(),
            'encoding': encoding.clone().detach(),
            'frequencies': self.w.clone().detach(),
        })
            
        return encoding
