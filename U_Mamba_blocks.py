import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import time


class U_Mamba_block(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand_factor: int = 2, d_conv: int = 4):
        super(U_Mamba_block, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = self.expand_factor * self.d_model

        #left input projection: splits into SSM input and gating/residual
        self.left_proj = nn.Linear(self.d_model, self.d_inner, bias=False)

        #depthwise casual conv
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, padding=0, groups=self.d_inner, bias=True)

        # Projections for SSM parameters from the input
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        #Learnable diagonal A matrix, log-initialized
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))

        #right path
        self.right_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.post_norm = nn.LayerNorm(self.d_inner)

        #output projection from inner dimension back to the model dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        #nicer init on linear maps
        nn.init.xavier_uniform_(self.left_proj.weight)
        nn.init.xavier_uniform_(self.right_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def ssm(self, x):
        # x has shape [B, L, d_inner]
        b, l, d_inner = x.shape
        #print(f"SSM input shape: {x.shape}")

        # Discretize A
        A = -F.softplus(self.A_log).unsqueeze(0).expand(self.d_inner, -1).to(x.dtype) # [d_inner, d_state]
        
        # Project x to get delta, B, and C
        delta = F.softplus(self.delta_proj(x)).to(x.dtype).clamp(min=1e-4, max=10) # [B, L, d_inner]
        B = self.B_proj(x) # [B, L, d_state]
        C = self.C_proj(x) # [B, L, d_state]

        #a more optimized implementation would use a parallel scan
        y_out = []
        h = x.new_zeros(b, self.d_inner, self.d_state)  # [B, d_inner, d_state]

        for i in range(l):
            # Get parameters for this timestep
            delta_t = delta[:, i, :] # [B, d_inner]
            B_t = B[:, i, :]       # [B, d_state]
            C_t = C[:, i, :]       # [B, d_state]
            x_t = x[:, i, :]         # [B, d_inner]

            # Discretize A for this timestep
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A) # [B, d_inner, d_state]
            
            # Discretize B and apply input x
            delta_B_x = (delta_t * x_t).unsqueeze(-1) * B_t.unsqueeze(1)  
            
            # State update
            h = A_bar * h + delta_B_x
            if self.training:
                h = F.dropout(h, p=0.05, training=True)
            
            # Output for this timestep
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            y_out.append(y_t)

        y_out = torch.stack(y_out, dim=1) # [B, L, d_inner]
        return y_out
    
    def forward(self, x):
        #x has shape [B, L, D] where L is sequence length, D is d_model
        b, l, d_model = x.shape

        #left path
        left_branch = self.left_proj(x).transpose(1, 2)  # [B, d_inner, L]
        left_branch = F.pad(left_branch, (self.d_conv - 1, 0))  # left pad only
        left_branch = self.conv1d(left_branch) #set conv1d padding to 0
        left_branch = left_branch.transpose(1, 2)  # [B, L, d_inner]
        left_branch = F.silu(left_branch)  

        #get ssm output
        ssm_out = self.ssm(left_branch)

        #right path
        right_branch_gate = self.right_proj(x) # [B, L, d_inner]
        right_branch_gate = F.silu(right_branch_gate) 
        right_branch_gate = F.dropout(right_branch_gate, p=0.1, training=self.training)

        #combine paths
        out = ssm_out * right_branch_gate
        out = self.post_norm(out)
        out = self.out_proj(out)  # [B, L, d_model]
        return out


class ConvResidual3D(nn.Module):
    def __init__(self, in_channels):
        super(ConvResidual3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return x + self.block(x)
    
class Full_U_Mamba_Block(nn.Module):
    def __init__(self, channels: int, ssm_dim: int, use_residual: bool = True, use_checkpointing: bool = False):
        super(Full_U_Mamba_Block, self).__init__()
        self.use_residual = use_residual
        self.channels = channels
        self.use_checkpointing = use_checkpointing

        #2 conv residual blocks
        self.conv_residual1 = ConvResidual3D(channels)
        self.conv_residual2 = ConvResidual3D(channels)

        #U_Mamba block
        self.norm = nn.LayerNorm(channels)
        self.mamba_block = U_Mamba_block(d_model=channels, d_state=ssm_dim)

        self.res_scale = nn.Parameter(torch.tensor(0.1)) #start small, learn up

        #learned axis weights for combing scans (H/W/D)
        self.axis_w = nn.Parameter(torch.zeros(3))

    def _forward_impl(self, x):
        """The actual implementation of the forward pass."""
        identity = x
        B, C, H, W, D_vol = x.shape
        
        x = self.conv_residual1(x)
        x = self.conv_residual2(x)
        
        # Axis-wise Mamba application
        x_h = x.permute(0, 3, 4, 2, 1).reshape(B * W * D_vol, H, C)
        x_h_norm = self.norm(x_h)
        mamba_h_out = self.mamba_block(x_h_norm)
        mamba_h_out = mamba_h_out.reshape(B, W, D_vol, H, C).permute(0, 4, 3, 1, 2)

        x_w = x.permute(0, 2, 4, 3, 1).reshape(B * H * D_vol, W, C)
        x_w_norm = self.norm(x_w)
        mamba_w_out = self.mamba_block(x_w_norm)
        mamba_w_out = mamba_w_out.reshape(B, H, D_vol, W, C).permute(0, 4, 1, 3, 2)
        
        x_d = x.permute(0, 2, 3, 4, 1).reshape(B * H * W, D_vol, C)
        x_d_norm = self.norm(x_d)
        mamba_d_out = self.mamba_block(x_d_norm)
        mamba_d_out = mamba_d_out.reshape(B, H, W, D_vol, C).permute(0, 4, 1, 2, 3)

        #weighted combine instead of plain mean
        w = F.softmax(self.axis_w, dim=0)
        out = (w[0] * mamba_h_out + w[1] * mamba_w_out + w[2] * mamba_d_out)

        if self.use_residual:
            out = identity + self.res_scale * out
        
        return out

    def forward(self, x):
        if self.training and self.use_checkpointing:
            # Use checkpointing during training if enabled
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            # Standard forward pass for inference or if checkpointing is off
            return self._forward_impl(x)

""" model = Full_U_Mamba_Block(channels=64, ssm_dim=128)
x = torch.randn(2, 64, 16, 16, 16)  # (B, C, H, W, D)
y = model(x)
print(y.shape) """

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))

#strided convolution
class Strided_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Strided_Conv, self).__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, x):
        return self.down(x)
    
#transposed convolution
class Transposed_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transposed_Conv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)
