import torch
from torch import nn
from quant_transformer.quantization import QuantizedModule, Quantizer


class QuantizedLayerNorm(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.layernorm = org_module
        if self.qoutput:
            self.layernorm_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states, observation_mask=None):
        hidden_states = self.layernorm(hidden_states)
        if self.qoutput:
            hidden_states = self.layernorm_post_act_fake_quantize(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedSplitLayerNorm(QuantizedModule):

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.layernorm = nn.LayerNorm(org_module.normalized_shape, elementwise_affine=False)
        tmp = (org_module.bias.data / org_module.weight.data).detach().clone()
        self.bias = torch.nn.Parameter(tmp)
        if self.qoutput:
            self.layernorm_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states, observation_mask=None):
        hidden_states = self.layernorm(hidden_states)
        hidden_states += self.bias
        if self.qoutput:
            hidden_states = self.layernorm_post_act_fake_quantize(hidden_states, observation_mask, 1)
        return hidden_states


class GammaResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul_gamma = False

    def set_gamma(self, gamma):
        self.mul_gamma = True
        self.gamma = torch.nn.Parameter(gamma.data.detach().clone())

    def forward(self, input, hidden_states):
        if self.mul_gamma:
            input = input * self.gamma
        return input + hidden_states
