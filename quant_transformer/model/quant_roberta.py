"""PyTorch Quantized RoRobertaa model. """
# TODO: relative keys
# TODO: remove all the decoder, for some tasks, we need to add it.
# TODO: remove the past-key value
import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    ModelOutput,  # noqa: F401
    add_code_sample_docstrings,  # noqa: F401
    add_start_docstrings,  # noqa: F401
    add_start_docstrings_to_model_forward,  # noqa: F401
    replace_return_docstrings,  # noqa: F401
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,  # noqa: F401
    MaskedLMOutput,  # noqa: F401
    MultipleChoiceModelOutput,  # noqa: F401
    NextSentencePredictorOutput,  # noqa: F401
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,  # noqa: F401
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    ModuleUtilsMixin,
)
import logging
from quant_transformer.quantization import QuantizedModule
from quant_transformer.quantization import Quantizer
from .util_layernorm import GammaResidual, QuantizedLayerNorm
logger = logging.getLogger(__name__)


# backend could be chosen from 'academic' and 'trt'
class QuantizedRobertaEmbeddings(QuantizedModule):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.word_embeddings = Quantizer(org_module.word_embeddings, w_qconfig)
        self.position_embeddings = Quantizer(org_module.position_embeddings, w_qconfig)
        self.padding_idx = org_module.padding_idx
        self.token_type_embeddings = Quantizer(org_module.token_type_embeddings, w_qconfig)
        self.dropout = org_module.dropout
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = org_module.position_embedding_type
        self.register_buffer("position_ids", org_module.position_ids.clone())
        self.register_buffer(
            "token_type_ids",
            org_module.token_type_ids.clone(),
            persistent=False,
        )
        self.LayerNorm = QuantizedLayerNorm(org_module.LayerNorm, w_qconfig, a_qconfig,
                                            qoutput=qoutput, backend=backend)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        observation_mask=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings, observation_mask)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedRobertaSelfAttention(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.num_attention_heads = org_module.num_attention_heads
        self.attention_head_size = org_module.attention_head_size
        self.all_head_size = org_module.all_head_size
        # q, k, v is a linear layer, respectively
        self.query = Quantizer(org_module.query, w_qconfig)
        self.key = Quantizer(org_module.key, w_qconfig)
        self.value = Quantizer(org_module.value, w_qconfig)

        self.dropout = org_module.dropout
        self.position_embedding_type = org_module.position_embedding_type
        # TODO: distance embedding here, I don't understand it right now
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            raise NotImplementedError('current branch of computation is not supported yet')
            self.max_position_embeddings = org_module.max_position_embeddings
            self.distance_embedding = QuantizedRobertaEmbeddings(org_module.distance_embedding, w_qconfig, a_qconfig, qoutput=True)

        # activation quantizer here
        self.query_permute_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.key_transpose_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.value_permute_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.attention_probs_post_act_fake_quantize = Quantizer(None, a_qconfig)
        if self.qoutput:
            self.context_view_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def transpose_for_scores(self, x):
        # cut up for multi-heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        observation_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = self.query_permute_post_act_fake_quantize(query_layer, observation_mask, 2)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, self.key_transpose_post_act_fake_quantize(key_layer.transpose(-1, -2), observation_mask, 3))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError('the relative key is not supported yet')
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs = self.attention_probs_post_act_fake_quantize(attention_probs, observation_mask, 2)
        value_layer = self.value_permute_post_act_fake_quantize(value_layer, observation_mask, 2)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.qoutput:
            context_layer = self.context_view_post_act_fake_quantize(context_layer, observation_mask, 1)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class QuantizedRobertaSelfOutput(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        self.dense = Quantizer(org_module.dense, w_qconfig)
        self.dropout = org_module.dropout
        if self.backend == 'tensorrt':
            self.output_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.before_LayerNorm_residual = GammaResidual()
        self.LayerNorm = QuantizedLayerNorm(org_module.LayerNorm, w_qconfig, a_qconfig,
                                            qoutput=qoutput, backend=backend)

    def forward(self, hidden_states, input_tensor, observation_mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.backend == 'tensorrt':
            hidden_states = self.output_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_LayerNorm_residual(input_tensor, hidden_states)
        hidden_states = self.LayerNorm(hidden_states, observation_mask)
        return hidden_states


class QuantizedRobertaAttention(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        self.self = QuantizedRobertaSelfAttention(org_module.self, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        self.output = QuantizedRobertaSelfOutput(org_module.output, w_qconfig, a_qconfig, qoutput=self.qoutput, backend=self.backend)
        self.pruned_heads = org_module.pruned_heads

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        observation_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            observation_mask=observation_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states, observation_mask=observation_mask)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class QuantizedRobertaIntermediate(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        # dense is a linear layer
        self.dense = Quantizer(org_module.dense, w_qconfig)
        self.intermediate_act_fn = org_module.intermediate_act_fn
        if qoutput:
            self.intermediate_act_fn_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states, observation_mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if self.qoutput:
            hidden_states = self.intermediate_act_fn_post_act_fake_quantize(hidden_states, observation_mask, 1)
        return hidden_states


class QuantizedRobertaOutput(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend)
        self.qoutput = qoutput
        # dense is a linear
        self.dense = Quantizer(org_module.dense, w_qconfig)
        self.dropout = org_module.dropout
        if self.backend == 'tensorrt':
            self.output_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.before_LayerNorm_residual = GammaResidual()
        self.LayerNorm = QuantizedLayerNorm(org_module.LayerNorm, w_qconfig, a_qconfig,
                                            qoutput=qoutput, backend=backend)

    def forward(self, hidden_states, input_tensor, observation_mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.backend == 'tensorrt':
            hidden_states = self.output_post_act_fake_quantize(hidden_states, observation_mask, 1)
        hidden_states = self.before_LayerNorm_residual(input_tensor, hidden_states)
        hidden_states = self.LayerNorm(hidden_states, observation_mask)
        return hidden_states


class QuantizedRobertaLayer(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.chunk_size_feed_forward = org_module.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = QuantizedRobertaAttention(org_module.attention, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        self.intermediate = QuantizedRobertaIntermediate(org_module.intermediate, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        self.output = QuantizedRobertaOutput(org_module.output, w_qconfig, a_qconfig, qoutput=self.qoutput, backend=self.backend)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        observation_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            observation_mask=observation_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
            observation_mask,
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, observation_mask=None):
        intermediate_output = self.intermediate(attention_output, observation_mask=observation_mask)
        layer_output = self.output(intermediate_output, attention_output, observation_mask=observation_mask)
        return layer_output


class QuantizedRobertaEncoder(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        self.config = org_module.config
        self.layer = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            if i != self.config.num_hidden_layers - 1:
                self.layer.append(QuantizedRobertaLayer(org_module.layer[i], w_qconfig, a_qconfig, qoutput=True, backend=self.backend))
            else:
                self.layer.append(QuantizedRobertaLayer(org_module.layer[i], w_qconfig, a_qconfig, qoutput=self.qoutput, backend=self.backend))
        self.gradient_checkpointing = org_module.gradient_checkpointing

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        observation_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    observation_mask
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    observation_mask=observation_mask,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QuantizedRobertaPooler(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.qoutput = qoutput
        # dense is a linear layer
        self.getitem_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.dense = Quantizer(org_module.dense, w_qconfig)
        self.activation = org_module.activation
        if qoutput:
            self.pooler_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        qfirst_token_tensor = self.getitem_post_act_fake_quantize(first_token_tensor)
        pooled_output = self.dense(qfirst_token_tensor)
        pooled_output = self.activation(pooled_output)
        if self.qoutput:
            pooled_output = self.pooler_post_act_fake_quantize(pooled_output)
        return pooled_output


class QuantizedRobertaModel(QuantizedModule, ModuleUtilsMixin):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic'):
        super().__init__(backend=backend)
        self.config = org_module.config
        self.qoutput = qoutput
        self.embeddings = QuantizedRobertaEmbeddings(org_module.embeddings, w_qconfig, a_qconfig, qoutput=True, backend=self.backend)
        if org_module.pooler is not None:
            self.encoder = QuantizedRobertaEncoder(org_module.encoder, w_qconfig, a_qconfig, qoutput=False, backend=self.backend)
            self.pooler = QuantizedRobertaPooler(org_module.pooler, w_qconfig, a_qconfig, qoutput=self.qoutput, backend=self.backend)

        else:
            self.encoder = QuantizedRobertaEncoder(org_module.encoder, w_qconfig, a_qconfig, qoutput=self.qoutput, backend=self.backend)
            self.pooler = None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        observation_mask=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # it looks like this [batch_size, head, from_seq_length, to_seq_length]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            observation_mask=observation_mask,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask
        )
        sequence_output = encoder_outputs[0]
        # pooler is combined with linear and tanh
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class QuantizedRobertaClassificationHead(QuantizedModule):
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

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.getitem_post_act_fake_quantize(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.dropout_post_act_fake_quantize(x)
        x = self.out_proj(x)
        if self.qoutput:
            x = self.out_proj_post_act_fake_quantize(x)
        return x


class QuantizedRobertaForSequenceClassification(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self.is_remove_padding = is_remove_padding
        self.num_labels = org_module.num_labels
        self.config = org_module.config
        self.qoutput = qoutput
        self.roberta = QuantizedRobertaModel(org_module=org_module.roberta, w_qconfig=w_qconfig, a_qconfig=a_qconfig,
                                             qoutput=False, backend=self.backend)
        self.classifier = QuantizedRobertaClassificationHead(org_module.classifier,
                                                             w_qconfig=w_qconfig, a_qconfig=a_qconfig,
                                                             qoutput=self.qoutput, backend=self.backend)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_remove_padding:
            observation_mask = attention_mask.sum(1)
        else:
            observation_mask = None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QuantizedRobertaForQuestionAnswering(QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qoutput=True, backend='academic', is_remove_padding=False):
        super().__init__(backend)
        self.is_remove_padding = is_remove_padding
        self.config = org_module.config
        self.qoutput = qoutput
        self.num_labels = org_module.config.num_labels
        self.roberta = QuantizedRobertaModel(org_module=org_module.roberta, w_qconfig=w_qconfig, a_qconfig=a_qconfig, qoutput=True, backend=self.backend)
        self.qa_outputs = Quantizer(org_module.qa_outputs, w_qconfig)
        if self.qoutput:
            self.qa_outputs_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_remove_padding:
            observation_mask = attention_mask.sum(1)
        else:
            observation_mask = None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observation_mask=observation_mask,
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
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
