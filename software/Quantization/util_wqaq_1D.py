import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function

# ********************* range_trackers *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedErrorF

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':
            min_val = torch.min(torch.min(input, 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(input, 2, keepdim=True)[0], 1, keepdim=True)[0]            
        self.update_range(min_val, max_val)
        
class GlobalRangeTracker(RangeTracker):
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))
class AveragedRangeTracker(RangeTracker):
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)
class LinearWeightRangeTracker(RangeTracker):
    def __init__(self,q_level):
        super().__init__(q_level)
        self.register_buffer('min_val',torch.zeros(1))
        self.register_buffer('max_val',torch.zeros(1))
        self.register_buffer('first_l', torch.zeros(1))
        
    def update_range(self, min_val, max_val):
        self.min_val=min_val
        self.max_val=max_val       
# ********************* quantizers*********************
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

    def update_params(self):
        raise NotImplementedError

    def quantize(self, input):
        output = input / self.scale + self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    def clamp(self, input):
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    def dequantize(self, input):
        output = (input - self.zero_point) * self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.update_params()
            output = self.quantize(input)
            output = self.round(output)
            output = self.clamp(output)
            output = self.dequantize(output)
        return output
class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1))
class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))

class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))
        self.scale = quantized_range / float_range
        self.zero_point = torch.zeros_like(self.scale)

class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        quantized_range = self.max_val - self.min_val
        float_range = self.range_tracker.max_val - self.range_tracker.min_val
        self.scale = float_range/quantized_range
        self.zero_point = torch.round(self.max_val -  self.range_tracker.max_val / self.scale)

# ********************* quantify the convolution operation *********************
class Conv1d_Q(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        a_bits=8,
        w_bits=8,
        q_type=1,
        first_layer=0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight) 

        output = F.conv1d(
            input=q_input,
            weight=q_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return output

def reshape_to_activation(input):
    return input.reshape(1, -1, 1)
def reshape_to_weight(input):
    return input.reshape(-1, 1, 1)
def reshape_to_bias(input):
    return input.reshape(-1)


# ********************* quantify the full connection layer *********************
class Linear_Q(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        a_bits =16,
        w_bits =8,
        q_type = 1,
        ):
        super(Linear_Q,self).__init__(
            in_features = in_features,
            out_features = out_features,
            bias = bias
            )
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits = a_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits = w_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits = a_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits = w_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))

    def forward(self, input):
        input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(self.weight) 
        output = F.linear(
            input = q_input,
            weight = q_weight,
            bias = self.bias
            )
        return output

class AvgPool1d_Q(nn.AvgPool1d):
    def __init__(
        self,
        kernel_size,
        stride,
        padding = 0,
        a_bits =16,
        q_type = 1,
        ):
        super(AvgPool1d_Q,self).__init__(
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
            )
    
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits = a_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))
           
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits = a_bits, range_tracker=LinearWeightRangeTracker(q_level='L'))

    def forward(self, input):
        input = self.activation_quantizer(input)
        q_input = input
        output = F.avg_pool1d(q_input, kernel_size=2, stride=2)
        return output

# *********************BN fusion and quantization *********************
class BNFold_Conv1d_Q(Conv1d_Q):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-5,
        momentum=0.01,
        a_bits=16,
        w_bits=8,
        q_type=1,
        first_layer=0
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
            self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=GlobalRangeTracker(q_level='C', out_channels=out_channels))
        self.first_layer = first_layer

    def forward(self, input):
        if self.training:
            output = F.conv1d(
                input=input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            dims = [dim for dim in range(3) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)

            if self.bias is not None:##即bias为True
                bias = reshape_to_bias(self.beta + (self.bias -  batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
                weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(batch_var + self.eps))
                
            else:
                bias = reshape_to_bias(self.beta - batch_mean  * (self.gamma / torch.sqrt(batch_var + self.eps)))
                weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(batch_var + self.eps))
        else:
            if self.bias is not None:#True
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))
            else:
                bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))

        if not self.first_layer:
            input = self.activation_quantizer(input)
        q_input = input
        q_weight = self.weight_quantizer(weight) 
        if self.training:
            output = F.conv1d(
              input=q_input,
              weight=q_weight,
              bias = bias,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
        else:
            output = F.conv1d(
              input=q_input,
              weight=q_weight,
              bias = bias,
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
        return output
       
class QuanConv1d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, last_relu=0, abits=8, wbits=8, bn_fold=0, q_type=1, first_layer=0):
        super(QuanConv1d, self).__init__()
        self.last_relu = last_relu
        self.bn_fold = bn_fold
        self.first_layer = first_layer

        if self.bn_fold == 1:
            self.bn_q_conv = BNFold_Conv1d_Q(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, q_type=q_type, first_layer=first_layer)
        else:
            self.q_conv = Conv1d_Q(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, a_bits=abits, w_bits=wbits, q_type=q_type, first_layer=first_layer)
            self.bn = nn.BatchNorm1d(output_channels, momentum=0.01) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.first_layer:
            x = self.relu(x)
        if self.bn_fold == 1:
            x = self.bn_q_conv(x)
        else:
            x = self.q_conv(x)
            x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x
