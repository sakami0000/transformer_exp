import torch
from torch import nn
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    ACT2FN,
    BertEncoder,
    BertAttention,
    BertLayer,
)


class EncoderEmbeddings(nn.Module):
    """Construct the embeddings from Exercise ID, Exercise category, and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.id_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.category_embeddings = nn.Embedding(
            config.category_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )

        self.linear_embed = nn.Linear(config.embedding_size * 3, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        category_ids=None,
        position_ids=None,
        timestamp=None,
        elapsed_time=None,
        inputs_embeds=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.id_embeddings(input_ids)

        category_embeddings = self.category_embeddings(category_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = torch.cat(
            [inputs_embeds, category_embeddings, position_embeddings], dim=-1
        )
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransposeBatchNorm1d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_ch)

    def forward(self, x):
        return self.norm(x.transpose(2, 1)).transpose(2, 1)


class DecoderEmbeddings(nn.Module):
    """Construct the embeddings from Response and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # response ids
        #   0: padding id
        #   1: start embedding
        #   3: incorrect answer
        #   4: correct answer
        self.response_embeddings = nn.Embedding(
            4, config.response_embedding_size, padding_idx=0
        )
        self.numerical_embeddings = nn.Sequential(
            TransposeBatchNorm1d(2), nn.Linear(2, config.embedding_size)
        )
        self.elapsed_time_embeddings = nn.Embedding(
            config.max_elapsed_seconds + 2, config.embedding_size, padding_idx=0
        )
        self.lag_time_embeddings = nn.Embedding(
            int(config.max_lag_minutes / 10) + 7, config.embedding_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )

        self.linear_embed = nn.Linear(
            config.response_embedding_size + config.embedding_size * 4,
            config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_lag_time(self, timestamp: torch.LongTensor) -> torch.FloatTensor:
        unique_time, inverse_indices = torch.unique_consecutive(
            timestamp, return_inverse=True
        )
        inverse_indices = torch.max(
            inverse_indices.min(dim=1)[0].unsqueeze(1), inverse_indices - 1
        )
        lag_time = (timestamp - unique_time[inverse_indices]).float() / (
            1000 * 60
        )  # minutes
        return lag_time.clamp(min=0, max=self.config.max_lag_minutes)

    def forward(
        self,
        input_ids=None,
        category_ids=None,
        position_ids=None,
        timestamp=None,
        elapsed_time=None,
        inputs_embeds=None,
    ):
        response_ids = input_ids
        response_embeddings = self.response_embeddings(response_ids)

        # numerical features
        lag_time_num = self.get_lag_time(timestamp)
        elapsed_time_num = elapsed_time.clamp(
            min=0, max=self.config.max_elapsed_seconds
        )

        numerical_states = torch.stack([lag_time_num.log1p(), elapsed_time_num], dim=-1)
        numerical_embeddings = self.numerical_embeddings(numerical_states)

        # lag time as categorical embedding
        lag_time_cat = torch.where(
            lag_time_num < 6, lag_time_num.long(), ((lag_time_num - 1) / 10).long() + 6
        )
        lag_time_embeddings = self.lag_time_embeddings(lag_time_cat)

        # elapsed time as categorical embedding
        elapsed_time_cat = (elapsed_time.long() + 1).clamp(
            min=0, max=self.config.max_elapsed_seconds
        )
        elapsed_time_embeddings = self.elapsed_time_embeddings(elapsed_time_cat)

        # positional embedding
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = torch.cat(
            [
                response_embeddings,
                numerical_embeddings,
                lag_time_embeddings,
                elapsed_time_embeddings,
                position_embeddings,
            ],
            dim=-1,
        )
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SaintSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class SaintAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = SaintSelfOutput(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], residual)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class SaintIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SaintOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class SaintLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = SaintAttention(config)
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = SaintAttention(config)
        self.intermediate = SaintIntermediate(config)
        self.output = SaintOutput(config)


class SaintEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [SaintLayer(config) for _ in range(config.num_hidden_layers)]
        )


class SaintModel(BertPreTrainedModel):
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

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.is_decoder:
            self.embeddings = DecoderEmbeddings(config)
        else:
            self.embeddings = EncoderEmbeddings(config)

        self.encoder = SaintEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        timestamp=None,
        category_ids=None,
        position_ids=None,
        elapsed_time=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
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
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Upper Triangular Mask
        attention_mask = torch.tril(
            torch.matmul(attention_mask[:, :, None], attention_mask[:, None, :])
        )  # [batch_size, seq_length, seq_length]

        if category_ids is None:
            category_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            # Upper Triangular Mask
            encoder_attention_mask = torch.tril(
                torch.matmul(
                    encoder_attention_mask[:, :, None],
                    encoder_attention_mask[:, None, :],
                )
            )  # [batch_size, seq_length, seq_length]

            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        if position_ids is None:
            position_ids = (1 - (timestamp <= timestamp.roll(1, dims=1)).long()).cumsum(
                dim=1
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            category_ids=category_ids,
            position_ids=position_ids,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        return encoder_output


class SaintEncoderDecoderModel(nn.Module):
    """SAINT-based model

    References
    ----------
    - https://arxiv.org/abs/2002.07033
    - https://arxiv.org/abs/2010.12042
    """

    def __init__(self, encoder_config, decoder_config, num_labels: int = 1):
        super().__init__()
        self.encoder = SaintModel(encoder_config)
        self.decoder = SaintModel(decoder_config)

        self.dropout = nn.Dropout(decoder_config.hidden_dropout_prob)
        self.classifier = nn.Linear(decoder_config.hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        timestamp=None,
        category_ids=None,
        elapsed_time=None,
        response_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        **kwargs,
    ):
        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }
        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timestamp=timestamp,
                category_ids=category_ids,
                inputs_embeds=inputs_embeds,
                **kwargs_encoder,
            )

        # Decode
        decoder_output = self.decoder(
            input_ids=response_ids,
            attention_mask=decoder_attention_mask,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            **kwargs_decoder,
        )

        decoder_output = self.dropout(decoder_output.last_hidden_state)
        logits = self.classifier(decoder_output).squeeze(2)
        return logits
