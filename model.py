import torch
import torch.nn as nn

from model_parts import _conv_, _context_block_, _upsample_, _residual_group_, _ResBlock_CBAM_

class SAR_CAM(nn.Module) :
    def __init__(self, scale, in_channels, channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(SAR_CAM, self).__init__()

        self._conv_in_ = _conv_(in_channels, channels, kernel_size, stride, dilation, bias)
        self._conv_out_ = _conv_(channels, in_channels, kernel_size, stride, dilation, bias)
        self._down_ = nn.MaxPool2d(kernel_size = scale, stride = scale)
        self._cb_ = _context_block_(channels, kernel_size, stride, dilation, bias)
        self._rg_1_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._rg_2_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._rg_3_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._conv_ = _conv_(channels, channels, kernel_size, stride, dilation, bias)
        self._rc_ = _ResBlock_CBAM_(channels, kernel_size, stride, dilation, bias)
        self._up_ = _upsample_(scale, channels, kernel_size, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        skip_connection = out
        out = self._down_(out)
        out = self._cb_(out)
        out = self._rg_1_(out)
        concat_1 = out
        out = self._rg_2_(out)
        concat_2 = out
        out = self._rg_3_(out)
        concat_3 = out
        out = self._conv_(concat_1 + concat_2 + concat_3)
        concat_4 = self._rc_(out)
        out = torch.cat([concat_1, concat_2, concat_3, concat_4], dim = 1)
        out = self._up_(out, skip_connection)
        out = self._conv_out_(out)
        out = x + out

        return out

    def initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                # Apply Xavier Uniform Initialization
                torch.nn.init.xavier_uniform_(m.weight.data)

                if m.bias is not None :
                    m.bias.data.zero_()

def Model(scale, in_channels, channels, kernel_size, stride, dilation, bias) :
    return SAR_CAM(scale, in_channels, channels, kernel_size, stride, dilation, bias)