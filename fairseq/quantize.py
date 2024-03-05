from collections import namedtuple
import math
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)

logger = logging.getLogger(__name__)

def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        flatten_dims = (1, -1) if len(x.shape) == 2 else (0, -1) # determine flatten dims by the shape of x

        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
        scale = torch.max(scale, min_scale)

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad, flatten_dims=(1, -1))
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False):
    if qparams:
        if qparams.num_bits:
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace)

    return x


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True,
                  signed=False, stochastic=True):
    if qparams:
        if qparams.num_bits:
            return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                               stochastic)
    elif num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic)

    return x


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure, device='cuda'))
        self.register_buffer('running_range', torch.zeros(*shape_measure, device='cuda'))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1, device='cuda'))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.stride = stride

    def forward(self, input, num_bits, num_grad_bits):
        if num_bits == 0:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        weight_qparams = calculate_qparams(self.weight, num_bits=num_bits, flatten_dims=(1, -1),
                                           reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)

        qinput = self.quantize_input(input, num_bits)
        output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1))

        return output
        

    def conv2d_quant_act(self, input_fw, input_bw, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
                         error_bits=0, gc_bits=0):
        out1 = F.conv2d(input_fw, weight.detach(), bias.detach() if bias is not None else None,
                        stride, padding, dilation, groups)
        out2 = F.conv2d(input_bw.detach(), weight, bias,
                        stride, padding, dilation, groups)
        out1 = quantize_grad(out1, num_bits=error_bits)
        out2 = quantize_grad(out2, num_bits=gc_bits)
        return out1 + out2 - out2.detach()

class QLinear(nn.Linear):
    """
    A quantized linear layer for PyTorch.
    """

    def __init__(self, in_features, out_features, bias=True, num_bits=None, is_cyclic_precision=False, num_cyclic_period=None, cyclic_num_bits_schedule=None, log_every=None):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(1, -1))
        self.num_bits = num_bits

        self.is_cyclic_precision = is_cyclic_precision

        if self.is_cyclic_precision:
            self.forward_iters = 0
            self.num_cyclic_period = num_cyclic_period
            self.cyclic_num_bits_schedule = cyclic_num_bits_schedule
            self.log_every = log_every

    def forward(self, input):
        """
        Performs quantized linear transformation.

        Args:
            input: Input tensor.
            num_bits: Number of bits for quantization.

        Returns:
            Quantized output tensor.
        """
        if self.is_cyclic_precision:
            self.cyclic_adjust_precision(self.num_cyclic_period)
            # cyclic_period = int(self.forward_iters / self.num_cyclic_period)
            # if cyclic_period != 0:
            #     self.cyclic_adjust_precision(cyclic_period)
            # else:
            #     self.cyclic_adjust_precision(1)
        
            self.forward_iters += 1

        if self.num_bits is None or self.num_bits == 0:
            output = F.linear(input, self.weight, self.bias)
            return output

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
        else:
            qbias = None

        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits, flatten_dims=(1, -1), reduce_dim=None
        )
        qweight = quantize(self.weight, qparams=weight_qparams)

        qinput = self.quantize_input(input, self.num_bits)
        output = F.linear(qinput, qweight, qbias)

        return output

    def linear_quant_act(self, input, act_fn=torch.nn.functional.relu):
        """
        Performs quantized linear transformation followed by activation.

        Args:
            input: Input tensor.
            act_fn: Activation function to apply (default: ReLU).

        Returns:
            Quantized output tensor with activation applied.
        """

        output = self.forward(input)  # Perform quantized linear transformation

        # Quantize activations (assuming you have a 'quantize_act' function)
        if self.num_bits is not None and self.num_bits > 0:
            q_output = self.quantize_act(output, num_bits=self.num_bits)
        else:
            q_output = output

        # Apply activation function
        return act_fn(q_output)

    def cyclic_adjust_precision(self, cyclic_period):
        assert len(eval(self.cyclic_num_bits_schedule)) == 2
        
        num_bit_min, num_bit_max = eval(self.cyclic_num_bits_schedule)

        self.num_bits = np.rint(num_bit_min +
                                0.5 * (num_bit_max - num_bit_min) *
                                (1 + np.cos(np.pi * ((self.forward_iters % cyclic_period) / cyclic_period) + np.pi)))
        
        if self.forward_iters % self.log_every == 0:
            logger.info('Iter [{}] num_bits = {} cyclic precision'.format(self.forward_iters, self.num_bits))

def reset_forward_iters(model):
    for name, module in model.named_modules():
        if 'encoder' in name and isinstance(module, QLinear):
            module.forward_iters = 0

def quantize_model(model, num_bits, device='cpu', exclude=None, is_cyclic_precision=False, num_cyclic_period=None, cyclic_num_bits_schedule=None, log_every=None):
    """
    Quantizes weights and activations of Linear layers in a PyTorch model.

    Args:
        model: PyTorch model to be quantized.
        num_bits: Number of bits for weight quantization.
        device: Device to set tensors to.
        exclude: Layers to exclude from linear layer quantization. (ex. self_attn, fc)

    Returns:
        Quantized PyTorch model.
    """
    # store module names bc OrderedDict cannot be mutated
    to_be_quantized = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if exclude is not None and exclude in name:
                continue

            # clone original weights and bias for usage
            original_weight = module.weight.data.clone().to(device)
            original_bias = module.bias.data.clone().to(device) if module.bias is not None else None

            # quantize weights using QParams calculation
            weight_qparams = calculate_qparams(
                original_weight, num_bits=num_bits, reduce_dim=None
            )
            quantized_weight = quantize(original_weight, qparams=weight_qparams).to(device)

            # quantize weights
            quantized_bias = None
            if module.bias is not None:
                quantized_bias = quantize(original_bias, num_bits=num_bits).to(device)

            # append module name and quantized information for later replacement
            to_be_quantized.append((name, quantized_weight, quantized_bias if module.bias is not None else None))

    # quantize layers (swap nn.Linear for QLinear and swap weights for quantized weights)
    for name, quantized_weight, quantized_bias in to_be_quantized:
        if 'encoder' in name: # only quantize encoder
            layer_to_replace, last_attr_name = get_layer_to_replace(model, name.split('.'))
            setattr(layer_to_replace, last_attr_name, QLinear(quantized_weight.size(0), quantized_weight.size(1), num_bits=num_bits, is_cyclic_precision=is_cyclic_precision, num_cyclic_period=num_cyclic_period, cyclic_num_bits_schedule=cyclic_num_bits_schedule, log_every=log_every))

            new_layer = get_layer(model, name.split('.'))
            new_layer.weight.data = quantized_weight
            if quantized_bias is not None:
                new_layer.bias.data = quantized_bias

    return model

def get_layer_to_replace(model, attribute_names):
    curr_obj = model
    # Convert to indexing call
    for attr_name in attribute_names[:-1]:
        if isinstance(attr_name, int):
            curr_obj = curr_obj[attr_name]
        else:
            curr_obj = getattr(curr_obj, attr_name)

    return curr_obj, attribute_names[-1]

def get_layer(model, attribute_names):
    curr_obj = model
    # Convert to indexing call
    for attr_name in attribute_names:
        if isinstance(attr_name, int):
            curr_obj = curr_obj[attr_name]
        else:
            curr_obj = getattr(curr_obj, attr_name)

    return curr_obj

if __name__ == '__main__':
    x = torch.rand(2, 3).to('cuda')
    x_q = quantize(x, num_bits=8, dequantize=True)
    print(x)
    print(x_q)