import torch
from collections import OrderedDict
from quant_transformer.model.util_layernorm import GammaResidual, QuantizedLayerNorm, QuantizedSplitLayerNorm
import logging
logger = logging.getLogger("transformer")


def get_weight_modules(model, config_model):
    num_layer = model.config.num_hidden_layers
    weight_modules = []
    if config_model.model_type == 'bert' or config_model.model_type == 'roberta':
        for i in range(num_layer):
            layer = getattr(model, config_model.model_type).encoder.layer[i]
            # self attn, q, k, v
            weight_modules.append([
                layer.attention.self.query,
                layer.attention.self.key,
                layer.attention.self.value])
            # linear
            weight_modules.append([layer.intermediate.dense])
    if config_model.model_type == 'bart':
        # encoder
        for i in range(num_layer):
            layer = model.model.encoder.layers[i]
            # self attn q, k, v
            weight_modules.append([layer.self_attn.q_proj,
                                   layer.self_attn.k_proj,
                                   layer.self_attn.v_proj])
            # linear
            weight_modules.append([layer.fc1])
        # decoder
        for i in range(num_layer):
            layer = model.model.decoder.layers[i]
            # self attn q, k, v
            weight_modules.append([layer.self_attn.q_proj,
                                   layer.self_attn.k_proj,
                                   layer.self_attn.v_proj])
            # cross attn q
            weight_modules.append([layer.encoder_attn.q_proj])
            # linear
            weight_modules.append([layer.fc1])
    return weight_modules


@torch.no_grad()
def gamma_migration(model, config_quant, config_model):

    def replace_module(name, new_module):
        name = name.split('.')
        fa_module = model
        for t in name[: -1]:
            fa_module = getattr(fa_module, t)
        setattr(fa_module, name[-1], new_module)

    weight_modules = get_weight_modules(model, config_model)
    cnt = 0
    last_layernorm = None
    name2modules = OrderedDict(model.named_modules())
    for name, module in name2modules.items():
        if isinstance(module, GammaResidual):
            if last_layernorm is not None:
                old_module = last_layernorm[1]
                new_module = QuantizedSplitLayerNorm(old_module.layernorm,
                                                     config_quant.w_qconfig,
                                                     config_quant.a_qconfig,
                                                     old_module.qoutput,
                                                     old_module.backend).cuda().eval()
                replace_module(last_layernorm[0], new_module)
                module.set_gamma(old_module.layernorm.weight.data)
                for w in weight_modules[cnt]:
                    w.weight.data *= old_module.layernorm.weight.data.detach().clone()
                cnt += 1
                last_layernorm = None
        if isinstance(module, QuantizedLayerNorm):
            last_layernorm = [name, module]
    return model


def delay_ln(model, config_quant, config_model):
    model = gamma_migration(model, config_quant, config_model)
    return model
