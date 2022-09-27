"""PyTorch Quantized BART model. """
import random
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
import logging
from quant_transformer.quantization import QuantizedModule, Quantizer
from .util_layernorm import QuantizedLayerNorm, GammaResidual

logger = logging.getLogger(__name__)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class QuantizedBartLearnedPositionalEmbedding(QuantizedModule):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        super().__init__(backend=backend)
        self.offset = 2
        self.qoutput = qoutput
        num_embeddings, embedding_dim = list(org_module.parameters())[0].size()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.position_embeddings.load_state_dict(org_module.state_dict())
        self.position_embeddings = Quantizer(self.position_embeddings, w_qconfig)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.position_embeddings.weight.device
        )
        return self.position_embeddings(positions + self.offset)


class QuantizedBartAttention(QuantizedModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module,
        w_qconfig,
        a_qconfig,
        qoutput=True,
        backend='academic'
    ):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.embed_dim = org_module.embed_dim
        self.num_heads = org_module.num_heads
        self.dropout = org_module.dropout
        self.head_dim = org_module.head_dim

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = org_module.is_decoder

        self.k_proj = Quantizer(org_module.k_proj, w_qconfig)
        self.v_proj = Quantizer(org_module.v_proj, w_qconfig)
        self.q_proj = Quantizer(org_module.q_proj, w_qconfig)
        self.out_proj = Quantizer(org_module.out_proj, w_qconfig)

        # activation quantizer here
        self.query_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.key_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.value_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.attention_probs_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.context_post_act_fake_quantize = Quantizer(None, a_qconfig)
        if self.qoutput:
            self.out_proj_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        observation_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.query_post_act_fake_quantize(
            self.q_proj(hidden_states) * self.scaling, observation_mask, 1
        )
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(
                self.key_post_act_fake_quantize(self.k_proj(key_value_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_post_act_fake_quantize(self.v_proj(key_value_states), observation_mask, 1),
                -1,
                bsz
            )
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(
                self.key_post_act_fake_quantize(self.k_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_post_act_fake_quantize(self.v_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(
                self.key_post_act_fake_quantize(self.k_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )
            value_states = self._shape(
                self.value_post_act_fake_quantize(self.v_proj(hidden_states), observation_mask, 1),
                -1,
                bsz
            )

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`

            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = self.attention_probs_post_act_fake_quantize(attn_probs, observation_mask, 2)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.context_post_act_fake_quantize(attn_output, observation_mask, 1)
        attn_output = self.out_proj(attn_output)

        if self.qoutput:
            attn_output = self.out_proj_post_act_fake_quantize(attn_output, observation_mask, 1)

        return attn_output, attn_weights_reshaped, past_key_value


class QuantizedBartEncoderLayer(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        self.embed_dim = org_module.embed_dim
        self.self_attn = QuantizedBartAttention(org_module.self_attn, w_qconfig, a_qconfig,
                                                qoutput=False, backend=self.backend)
        self.before_self_attn_layer_norm_residual = GammaResidual()
        self.self_attn_layer_norm = QuantizedLayerNorm(
            org_module.self_attn_layer_norm, w_qconfig, a_qconfig,
            qoutput=True, backend=backend)
        self.dropout = org_module.dropout
        # fc is a linear layer
        self.fc1 = Quantizer(org_module.fc1, w_qconfig)
        self.activation_fn = org_module.activation_fn
        self.activation_dropout = org_module.activation_dropout
        self.fc1_act_fn_post_act_fake_quantize = Quantizer(None, a_qconfig)

        self.fc2 = Quantizer(org_module.fc2, w_qconfig)
        self.before_final_layer_norm_residual = GammaResidual()
        self.final_layer_norm = QuantizedLayerNorm(
            org_module.final_layer_norm, w_qconfig, a_qconfig,
            qoutput=qoutput, backend=backend)

        if self.backend == 'tensorrt':
            self.self_attn_post_act_fake_quantize = Quantizer(None, a_qconfig)
            self.fc2_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        observation_mask=None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # attn residual
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            observation_mask=observation_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.backend == 'tensorrt':
            hidden_states = self.self_attn_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_self_attn_layer_norm_residual(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states, observation_mask)
        # ffn residual
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc1_act_fn_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.backend == 'tensorrt':
            hidden_states = self.fc2_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_final_layer_norm_residual(residual, hidden_states)
        hidden_states = self.final_layer_norm(hidden_states, observation_mask)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class QuantizedBartDecoderLayer(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        self.embed_dim = org_module.embed_dim

        self.self_attn = QuantizedBartAttention(
            org_module.self_attn, w_qconfig, a_qconfig,
            qoutput=False, backend=self.backend)
        self.dropout = org_module.dropout
        self.before_self_attn_layer_norm_residual = GammaResidual()
        self.self_attn_layer_norm = QuantizedLayerNorm(
            org_module.self_attn_layer_norm, w_qconfig, a_qconfig,
            qoutput=True, backend=backend)

        self.encoder_attn = QuantizedBartAttention(
            org_module.encoder_attn, w_qconfig, a_qconfig,
            qoutput=False, backend=self.backend)
        self.before_encoder_attn_layer_norm_residual = GammaResidual()
        self.encoder_attn_layer_norm = QuantizedLayerNorm(
            org_module.encoder_attn_layer_norm, w_qconfig, a_qconfig,
            qoutput=True, backend=backend)
        self.fc1 = Quantizer(org_module.fc1, w_qconfig)
        self.activation_fn = org_module.activation_fn
        self.activation_dropout = org_module.activation_dropout
        self.fc1_act_fn_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.fc2 = Quantizer(org_module.fc2, w_qconfig)
        self.before_final_layer_norm_residual = GammaResidual()
        self.final_layer_norm = QuantizedLayerNorm(
            org_module.final_layer_norm, w_qconfig, a_qconfig,
            qoutput=self.qoutput, backend=backend)

        if self.backend == 'tensorrt':
            self.self_attn_post_act_fake_quantize = Quantizer(None, a_qconfig)
            self.encoder_attn_post_act_fake_quantize = Quantizer(None, a_qconfig)
            self.fc2_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        observation_mask=None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            observation_mask=observation_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.backend == 'tensorrt':
            hidden_states = self.self_attn_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_self_attn_layer_norm_residual(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states, observation_mask)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                observation_mask=observation_mask,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            if self.backend == 'tensorrt':
                hidden_states = self.encoder_attn_post_act_fake_quantize(hidden_states, observation_mask, 1)
            hidden_states = self.before_encoder_attn_layer_norm_residual(residual, hidden_states)
            hidden_states = self.encoder_attn_layer_norm(hidden_states, observation_mask)
            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc1_act_fn_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.backend == 'tensorrt':
            hidden_states = self.fc2_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_final_layer_norm_residual(residual, hidden_states)
        hidden_states = self.final_layer_norm(hidden_states, observation_mask)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuantizedBartClassificationHead(QuantizedModule):
    """Head for sentence-level classification tasks."""

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        self.getitem_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.dense = Quantizer(org_module.dense, w_qconfig)
        self.dropout = org_module.dropout
        self.dropout_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.out_proj = Quantizer(org_module.out_proj, w_qconfig)
        if self.qoutput:
            self.out_proj_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.getitem_post_act_fake_quantize(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dropout_post_act_fake_quantize(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        if self.qoutput:
            hidden_states = self.out_proj_post_act_fake_quantize(hidden_states)
        return hidden_states


class QuantizedBartEncoder(QuantizedModule):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.config = org_module.config

        self.dropout = org_module.dropout
        self.layerdrop = org_module.layerdrop

        self.padding_idx = org_module.padding_idx
        self.max_source_positions = org_module.max_source_positions
        self.embed_scale = org_module.embed_scale

        self.embed_tokens = Quantizer(org_module.embed_tokens, w_qconfig)
        self.embed_positions = QuantizedBartLearnedPositionalEmbedding(
            org_module.embed_positions, w_qconfig, a_qconfig,
            qoutput=False, backend=self.backend)
        self.layernorm_embedding = QuantizedLayerNorm(
            org_module.layernorm_embedding, w_qconfig, a_qconfig, qoutput=True, backend=backend)

        self.layers = nn.ModuleList()
        for i in range(self.config.encoder_layers):
            if i != self.config.encoder_layers - 1:
                self.layers.append(QuantizedBartEncoderLayer(org_module.layers[i], w_qconfig, a_qconfig,
                                                             qoutput=True, backend=self.backend))
            else:
                self.layers.append(QuantizedBartEncoderLayer(org_module.layers[i], w_qconfig, a_qconfig,
                                                             qoutput=self.qoutput, backend=self.backend))

        self.gradient_checkpointing = org_module.gradient_checkpointing

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # word embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # position embeddings
        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states, observation_mask)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs,
                                          observation_mask=observation_mask,
                                          output_attentions=output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        observation_mask=observation_mask,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class QuantizedBartDecoder(QuantizedModule, ModuleUtilsMixin):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.config = org_module.config

        self.dropout = org_module.dropout
        self.layerdrop = org_module.layerdrop
        self.padding_idx = org_module.padding_idx
        self.max_target_positions = org_module.max_target_positions
        self.embed_scale = org_module.embed_scale

        self.embed_tokens = Quantizer(org_module.embed_tokens, w_qconfig)
        self.embed_positions = QuantizedBartLearnedPositionalEmbedding(
            org_module.embed_positions, w_qconfig, a_qconfig,
            qoutput=False, backend=self.backend)
        self.layernorm_embedding = QuantizedLayerNorm(
            org_module.layernorm_embedding, w_qconfig, a_qconfig, qoutput=True, backend=backend)

        self.layers = nn.ModuleList()
        for i in range(self.config.decoder_layers):
            if i != self.config.decoder_layers - 1:
                self.layers.append(QuantizedBartDecoderLayer(org_module.layers[i], w_qconfig, a_qconfig,
                                                             qoutput=True, backend=self.backend))
            else:
                self.layers.append(QuantizedBartDecoderLayer(org_module.layers[i], w_qconfig, a_qconfig,
                                                             qoutput=self.qoutput, backend=self.backend))
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states, observation_mask)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs,
                                      output_attentions=output_attentions,
                                      use_cache=use_cache,
                                      observation_mask=observation_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    observation_mask=observation_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class QuantizedBartModel(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.config = org_module.config

        self.shared = Quantizer(org_module.shared, w_qconfig)

        self.encoder = QuantizedBartEncoder(org_module.encoder, w_qconfig, a_qconfig,
                                            qoutput=True, backend=self.backend)
        self.decoder = QuantizedBartDecoder(org_module.decoder, w_qconfig, a_qconfig,
                                            qoutput=self.qoutput, backend=self.backend)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None,
        decoder_observation_mask=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                observation_mask=observation_mask,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=decoder_observation_mask,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class QuantizedBartForConditionalGeneration(QuantizedModule, ModuleUtilsMixin, GenerationMixin):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self.is_remove_padding = is_remove_padding
        self.config = org_module.config
        self.qoutput = qoutput
        self.model = QuantizedBartModel(org_module.model, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        self.lm_head = Quantizer(org_module.lm_head, w_qconfig)
        self.register_buffer("final_logits_bias", org_module.final_logits_bias.clone())
        self.main_input_name = org_module.main_input_name

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_remove_padding and decoder_attention_mask is None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = observation_mask
        elif self.is_remove_padding and decoder_attention_mask is not None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = decoder_attention_mask.sum(1)
        else:
            observation_mask = None
            decoder_observation_mask = None

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
            decoder_observation_mask=decoder_observation_mask
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class QuantizedBartForSequenceClassification(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self.is_remove_padding = is_remove_padding
        self.qoutput = qoutput
        self.config = org_module.config
        self.model = QuantizedBartModel(org_module.model, w_qconfig, a_qconfig, qoutput=False, backend=self.backend)
        self.classification_head = QuantizedBartClassificationHead(
            org_module.classification_head,
            w_qconfig,
            a_qconfig,
            qoutput=self.qoutput,
            backend=self.backend
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_remove_padding and decoder_attention_mask is None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = observation_mask
        elif self.is_remove_padding and decoder_attention_mask is not None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = decoder_attention_mask.sum(1)
        else:
            observation_mask = None
            decoder_observation_mask = None

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
            decoder_observation_mask=decoder_observation_mask
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class QuantizedBartForQuestionAnswering(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self.is_remove_padding = is_remove_padding
        self.qoutput = qoutput
        self.num_labels = org_module.num_labels
        self.config = org_module.config

        self.model = QuantizedBartModel(org_module.model, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        self.qa_outputs = Quantizer(org_module.qa_outputs, w_qconfig)
        if self.qoutput:
            self.qa_outputs_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_remove_padding and decoder_attention_mask is None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = observation_mask
        elif self.is_remove_padding and decoder_attention_mask is not None:
            observation_mask = attention_mask.sum(1)
            decoder_observation_mask = decoder_attention_mask.sum(1)
        else:
            observation_mask = None
            decoder_observation_mask = None

        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
            decoder_observation_mask=decoder_observation_mask,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        if self.qoutput:
            logits = self.qa_outputs_post_act_fake_quantize(logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
