"""
FarNet-II: U-net containing ConvLSTM and attention modules, 
prepared to compute magnetograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# Double convolution
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv(x)
        return x  

# ConvLSTM
class ConvLSTM(nn.Module):

    def __init__(self, input_channel, num_filter, b_h_w, kernel_size=3, stride=1, padding=1,device='cpu'):
        super().__init__()
        self.device = device
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        
        self._batch_size, self._state_height, self._state_width = b_h_w
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True)
        self._input_channel = input_channel
        self._num_filter = num_filter

    def forward(self, inputs=None, states=None, seq_len=11,device='cpu'):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(device)

        else:
            h, c = states

        outputs = []

        for index in range(seq_len):

            if inputs is None:

                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(device)
            
            else:
                x = inputs[index, ...]

            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)
            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

# Inconv
class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        
        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):

        x = self.conv(x)
        return x 

# Down class
class down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):

        x = self.mpconv(x)
        return x 

# Up class
class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        # Herencia
        super(up, self).__init__()

        self.bilinear = bilinear
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):

        if (self.bilinear): 
            x1 = torch.nn.functional.interpolate(x1, scale_factor=2)

        diffY = x2.size()[2] - x1.size()[2] 
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) 
        return x

# Out convolution
class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):

        x = self.conv(x)
        return x

# Attention module
class AttentionBlock(nn.Module):

    def __init__(self, in_channels_x, in_channels_g, int_channels):

        super(AttentionBlock, self).__init__()

        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size = 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
    
    def forward(self, x, g):

        x1 = self.Wx(x)
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.psi(nn.ReLU()(x1 + g1))
        out = nn.Sigmoid()(out)
        return out*x

# Whole UNet
class UNet(nn.Module):

    def __init__(self, n_channels=1, n_classes=9, n_seq=11, n_hidden=64,device='cpu',batch=2):

        super(UNet, self).__init__()

        self.device = device
        self.batch = batch
        self.inc = inconv(n_channels, n_hidden)
        self.down1 = down(n_hidden, 2*n_hidden)
        self.down2 = down(2*n_hidden, 4*n_hidden)
        self.dropout1 = nn.Dropout(0.5)
        self.down3 = down(4*n_hidden, 8*n_hidden)
        self.down4 = down(8*n_hidden, 8*n_hidden)
        self.LSTM1 = ConvLSTM(input_channel=8*n_hidden, num_filter=8*n_hidden, b_h_w=(batch,9,7),device=device)
        self.LSTM1_inv = ConvLSTM(input_channel=8*n_hidden, num_filter=8*n_hidden, b_h_w=(batch,9,7),device=device)
        self.conv1 = inconv(n_seq*2,n_seq)
        self.att1 = AttentionBlock(8*n_hidden,8*n_hidden,int(4*n_hidden))
        self.up1 = up(16*n_hidden,4*n_hidden)
        self.dropout2 = nn.Dropout(0.5)
        self.LSTM2 = ConvLSTM(input_channel=4*n_hidden, num_filter=4*n_hidden, b_h_w=(batch,18,15),device=device)
        self.LSTM2_inv = ConvLSTM(input_channel=4*n_hidden, num_filter=4*n_hidden, b_h_w=(batch,18,15),device=device)
        self.conv2 = inconv(n_seq*2,n_seq)
        self.att2 = AttentionBlock(4*n_hidden,4*n_hidden,int(2*n_hidden))
        self.up2 = up(8*n_hidden,2*n_hidden)
        self.LSTM3 = ConvLSTM(input_channel=2*n_hidden, num_filter=2*n_hidden, b_h_w=(batch,36,30),device=device)
        self.LSTM3_inv = ConvLSTM(input_channel=2*n_hidden, num_filter=2*n_hidden, b_h_w=(batch,36,30),device=device)
        self.conv3 = inconv(n_seq*2,n_seq)
        self.att3 = AttentionBlock(2*n_hidden,2*n_hidden,n_hidden)
        self.up3 = up(4*n_hidden,n_hidden)
        self.LSTM4 = ConvLSTM(input_channel=n_hidden, num_filter=n_hidden, b_h_w=(batch,72,60),device=device)
        self.LSTM4_inv = ConvLSTM(input_channel=n_hidden, num_filter=n_hidden, b_h_w=(batch,72,60),device=device)
        self.conv4 = inconv(n_seq*2,n_seq)
        self.att4 = AttentionBlock(n_hidden,n_hidden,int(n_hidden/2))
        self.up4 = up(2*n_hidden,n_hidden)
        self.outc = outconv(n_hidden, n_classes)

    # Forward pass
    def forward(self, x):
        
        x = x[:,:,np.newaxis,:,:]
        x = rearrange(x,'B S C H W -> (B S) C H W')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_2 = self.dropout1(x3)
        x4 = self.down3(x3_2)
        x5 = self.down4(x4)
        x5 = rearrange(x5,'(B S) C H W -> B S C H W',B=self.batch)
        x5 = rearrange(x5,'B S C H W -> S B C H W')
        x5LSTM = self.LSTM1(inputs=x5, states=None,device=self.device)[0]
        x5_inv = torch.flip(x5,[0])
        x5_inv_LSTM = self.LSTM1_inv(inputs=x5_inv, states=None,device=self.device)[0]
        x5_inv_LSTM = torch.flip(x5_inv_LSTM,[0])
        x5_LSTM_glob = torch.cat((x5LSTM,x5_inv_LSTM))
        x5_LSTM_glob = rearrange(x5_LSTM_glob,'S B C H W -> B C S H W')
        x5_LSTM_glob = rearrange(x5_LSTM_glob,'B C S H W -> (B C) S H W')
        x5LSTM = self.conv1(x5_LSTM_glob)
        x5LSTM = rearrange(x5LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x5LSTM = rearrange(x5LSTM,'B C S H W -> B S C H W')
        x5LSTM = rearrange(x5LSTM,'B S C H W -> (B S) C H W')
        att1x5 = self.att1(x4,x5LSTM)
        upx5 = self.up1(x5LSTM,att1x5)
        upx5_2 = self.dropout2(upx5)
        upx5 = rearrange(upx5_2,'(B S) C H W -> B S C H W',B=self.batch)
        upx5 = rearrange(upx5,'B S C H W -> S B C H W')
        x4LSTM = self.LSTM2(inputs=upx5, states=None,device=self.device)[0]
        upx5_inv = torch.flip(upx5,[0])
        upx5_inv_LSTM = self.LSTM2_inv(inputs=upx5_inv, states=None,device=self.device)[0]
        upx5_inv_LSTM = torch.flip(upx5_inv_LSTM,[0])
        x4_LSTM_glob = torch.cat((x4LSTM,upx5_inv_LSTM))   
        x4_LSTM_glob = rearrange(x4_LSTM_glob,'S B C H W -> B C S H W')
        x4_LSTM_glob = rearrange(x4_LSTM_glob,'B C S H W -> (B C) S H W')
        x4LSTM = self.conv2(x4_LSTM_glob)
        x4LSTM = rearrange(x4LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x4LSTM = rearrange(x4LSTM,'B C S H W -> B S C H W',B=self.batch)
        x4LSTM = rearrange(x4LSTM,'B S C H W -> (B S) C H W')
        att2x4 = self.att2(x3,x4LSTM)
        upx4 = self.up2(x4LSTM,att2x4)
        upx4 = rearrange(upx4,'(B S) C H W -> B S C H W',B=self.batch)
        upx4 = rearrange(upx4,'B S C H W -> S B C H W')
        x3LSTM = self.LSTM3(inputs=upx4, states=None,device=self.device)[0]
        upx4_inv = torch.flip(upx4,[0])
        upx4_inv_LSTM = self.LSTM3_inv(inputs=upx4_inv, states=None,device=self.device)[0]
        upx4_inv_LSTM = torch.flip(upx4_inv_LSTM,[0])
        x3_LSTM_glob = torch.cat((x3LSTM,upx4_inv_LSTM))
        x3_LSTM_glob = rearrange(x3_LSTM_glob,'S B C H W -> B C S H W')
        x3_LSTM_glob = rearrange(x3_LSTM_glob,'B C S H W -> (B C) S H W')
        x3LSTM = self.conv3(x3_LSTM_glob)
        x3LSTM = rearrange(x3LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x3LSTM = rearrange(x3LSTM,'B C S H W -> B S C H W')
        x3LSTM = rearrange(x3LSTM,'B S C H W -> (B S) C H W')
        att3x3 = self.att3(x2,x3LSTM)
        upx3 = self.up3(x3LSTM,att3x3)
        upx3 = rearrange(upx3,'(B S) C H W -> B S C H W',B=self.batch)
        upx3 = rearrange(upx3,'B S C H W -> S B C H W')
        x2LSTM = self.LSTM4(inputs=upx3, states=None,device=self.device)[0]
        upx3_inv = torch.flip(upx3,[0])
        upx3_inv_LSTM = self.LSTM4_inv(inputs=upx3_inv, states=None,device=self.device)[0]
        upx3_inv_LSTM = torch.flip(upx3_inv_LSTM,[0])
        x2_LSTM_glob = torch.cat((x2LSTM,upx3_inv_LSTM))
        x2_LSTM_glob = rearrange(x2_LSTM_glob,'S B C H W -> B C S H W')
        x2_LSTM_glob = rearrange(x2_LSTM_glob,'B C S H W -> (B C) S H W')
        x2LSTM = self.conv4(x2_LSTM_glob) 
        x2LSTM = rearrange(x2LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x2LSTM = rearrange(x2LSTM,'B C S H W -> B S C H W') 
        x2LSTM = rearrange(x2LSTM,'B S C H W -> (B S) C H W')
        att4x2 = self.att4(x1,x2LSTM)
        upx2 = self.up4(x2LSTM,att4x2)
        x = self.outc(upx2)
        x = rearrange(x,'(B S) C H W -> B S C H W',B=self.batch)
        return x
