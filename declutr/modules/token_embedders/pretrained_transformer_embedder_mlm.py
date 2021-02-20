from typing import Any, Dict, Optional, Tuple, Union

import torch
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides
from transformers import AutoConfig, AutoModelForMaskedLM
from declutr.modules.transformer_encoder.bertencoder import BertEncoder, BertLayer, BertEmbeddings
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)


@TokenEmbedder.register("pretrained_transformer_mlm")
class PretrainedTransformerEmbedderMLM(PretrainedTransformerEmbedder):
    """
    This is a wrapper around `PretrainedTransformerEmbedder` that allows us to train against a
    masked language modelling objective while we are embedding text.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mlm".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    masked_language_modeling: `bool`, optional (default = `True`)
        If this is `True` and `masked_lm_labels is not None` in the call to `forward`, the model
        will be trained against a masked language modelling objective and the resulting loss will
        be returned along with the output tensor.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        masked_language_modeling: bool = True,
    ) -> None:
        TokenEmbedder.__init__(self)  # Call the base class constructor
        tokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        self.masked_language_modeling = masked_language_modeling

        if self.masked_language_modeling:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            # We only need access to the HF tokenizer if we are masked language modeling
            self.tokenizer = tokenizer.tokenizer
            # The only differences when masked language modeling are:
            # 1) `output_hidden_states` must be True to get access to token embeddings.
            # 2) We need to use `AutoModelForMaskedLM` to get the correct model
            self.transformer_model = AutoModelForMaskedLM.from_pretrained(
            # self.transformer_model = RobertaForAugment.from_pretrained()
                model_name, config=self.config, **(transformer_kwargs or {})
            )
        # Eveything after the if statement (including the else) is copied directly from:
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/pretrained_transformer_embedder.py
        else:
            from allennlp.common import cached_transformers

            self.transformer_model = cached_transformers.get(
                model_name, True, override_weights_file, override_weights_strip_prefix
            )
            self.config = self.transformer_model.config

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.encoder = BertEncoder(self.config)
        self.layer = torch.nn.ModuleList([BertLayer(self.config)
                                    for _ in range(self.config.num_hidden_layers)])
        self.embeddings = BertEmbeddings(self.config)
        self.output_hidden_states = self.config.output_hidden_states

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @overrides
    def forward(
        self,
        augment: int,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor, torch.Tensor], torch.Tensor]:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
        masked_lm_labels: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces]`.

        # Returns:

        If `self.masked_language_modeling`, returns a `Tuple` of the masked language modeling loss
        and a `torch.Tensor` of shape: `[batch_size, num_wordpieces, embedding_size]`. Otherwise,
        returns only the `torch.Tensor` of shape: `[batch_size, num_wordpieces, embedding_size]`.
        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}  # type: ignore
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids
        if masked_lm_labels is not None and self.masked_language_modeling:
            parameters["labels"] = masked_lm_labels

        masked_lm_loss = None

        # if augment == -1:
        transformer_output = self.transformer_model(**parameters)

        if self.output_hidden_states:
            # Even if masked_language_modeling is True, we may not be masked language modeling on
            # the current batch. Check if masked language modeling labels are present in the input.
            if "labels" in parameters:
                masked_lm_loss = transformer_output[0]

            if self._scalar_mix:
                embeddings = self._scalar_mix(transformer_output[-1][1:])
            else:
                embeddings = transformer_output[-1][-1]
        else:
            embeddings = transformer_output[0]

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        # print("embeddings of original @@@@@@@",augment, masked_lm_loss, )
        # print("embeddings of original @@@@@@@",augment, masked_lm_loss, embeddings)

        input_ids = parameters['input_ids']
        device = input_ids.device

        # print("augment val is **********",augment)
        if augment >=0 and augment < 13:
            # embeds = self.embeddings(input_ids=input_ids)
            embeds = transformer_output[-1][0]
            # print("transformer output embed shape is ", embeds2)
            # print("embeds is $$$$$$$$$$", embeds)
            # shape of roughly [4, 256, 768]
            masks = parameters['attention_mask']
            head_mask = [None] * self.config.num_hidden_layers

            if augment == 0:
                input_lens = torch.sum(masks, dim=1)
                input_embeds, extended_attention_mask = self.cutoff(embeds, input_lens, device, masks)

                encoder_outputs = self.encoder(
                    input_embeds,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )
                embeddings = encoder_outputs[1][-1]
            else:
                # add hiddenstates, attention mask, head_mask definition
                hidden_states = embeds
                extended_attention_mask = masks.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.transformer_model.parameters()).dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

                all_hidden_states = ()
                for index, layer_module in enumerate(self.layer):
                    if index != augment-1:
                        if self.output_hidden_states:
                            all_hidden_states = all_hidden_states + (hidden_states,)

                        layer_outputs = layer_module(
                            hidden_states, extended_attention_mask, head_mask[index])
                        hidden_states = layer_outputs[0]

                    else:
                        # print("index~~~~~~~~~~~~~~~~~~~", index)
                        input_lens = torch.sum(masks, dim=1)
                        input_hidden_states, extended_cut_mask = self.cutoff(hidden_states, input_lens, device, masks)

                        if self.output_hidden_states:
                            all_hidden_states = all_hidden_states + (input_hidden_states,)

                            layer_outputs = layer_module(
                                input_hidden_states, extended_cut_mask, head_mask[index]
                            )
                            hidden_states = layer_outputs[0]
                embeddings = hidden_states
    
        if augment >= 13:
            if augment >=26:
                layer_num = augment-26
            else: 
                layer_num = augment-13
                
            # print("PCA color augmentation start!!")
            # print("layer num is ~~~~", layer_num)
            embeds = transformer_output[-1][0]
            masks = parameters['attention_mask']
            head_mask = [None] * self.config.num_hidden_layers
            hidden_states = embeds
            extended_attention_mask = masks.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.transformer_model.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            if layer_num == 0:
                input_lens = torch.sum(masks, dim=1)
                if augment >=26:
                    # print("hidden state sum before cutoff", torch.sum(hidden_states))
                    hidden_states, extended_cut_mask = self.cutoff(hidden_states, input_lens, device, masks)
                input_embeds = self.PCAjitter(hidden_states, device)
                encoder_outputs = self.encoder(
                    input_embeds,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )
                # print("encoder outputs shape", encoder_outputs[1][-1].shape)
                embeddings = encoder_outputs[1][-1]
            
            else:
                all_hidden_states = ()
                for index, layer_module in enumerate(self.layer):
                    if index != layer_num-1:
                        if self.output_hidden_states:
                            all_hidden_states = all_hidden_states + (hidden_states,)

                        layer_outputs = layer_module(
                            hidden_states, extended_attention_mask, head_mask[index])
                        hidden_states = layer_outputs[0]

                    else:
                        input_lens = torch.sum(masks, dim=1)
                        input_hidden_states = []

                        if augment >=26:
                            # print("hidden state sum before cutoff", torch.sum(hidden_states))
                            hidden_states, extended_cut_mask = self.cutoff(hidden_states, input_lens, device, masks)
                            # print("hidden state sum after cutoff", hidden_states)
                        input_hidden_states = self.PCAjitter(hidden_states, device)
                        # print("hidden state sum after PCA", input_hidden_states)

                        if self.output_hidden_states:
                            all_hidden_states = all_hidden_states + (input_hidden_states,)
                            if augment >=26:
                                layer_outputs = layer_module(
                                    input_hidden_states, extended_cut_mask, head_mask[index]
                            )   
                            else:
                                layer_outputs = layer_module(
                                    input_hidden_states, extended_attention_mask, head_mask[index]
                                )
                            hidden_states = layer_outputs[0]
                # print("len of all_hidden_state ^^^^^^^^^^", len(all_hidden_states))
                embeddings = hidden_states

        # print("embeddings of changed is ", augment,embeddings)
        return masked_lm_loss, embeddings


    def cutoff (
        self, embeds: torch.Tensor , input_lens: int , device: int, masks: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = []
        input_masks = []
        # embeds shape -> 4 * 256 * 768
        # print("embes.shape~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", embeds.shape)
        for i in range(embeds.shape[0]):
            #  args.aug_cutoff_ratio need to be defined (0.1)
            cutoff_length = int(input_lens[i] * 0.1)
            start = int(torch.rand(1).to(device)* (input_lens[i] - cutoff_length))
            # print(input_lens[i], cutoff_length, start)
            cutoff_embed = torch.cat((embeds[i][:start],
                                    torch.zeros([cutoff_length, embeds.shape[-1]],
                                                dtype=torch.float).to(device),
                                    embeds[i][start + cutoff_length:]), dim=0)
            cutoff_mask = torch.cat((masks[i][:start],
                                    torch.zeros([cutoff_length], dtype=torch.long).to(device),
                                    masks[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        # TODO: get_logits_from_embedding_output define
        # cutoff_outputs = self.transformer_model.get_logits_from_embedding_output(embedding_output=input_embeds,
        #                                                         attention_mask=input_masks, labels=labels
        # outputs = self.roberta.get_bert_output(input_embeds, attention_mask=attention_mask)

        extended_attention_mask = input_masks[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.transformer_model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return input_embeds, extended_attention_mask

    def PCAjitter(
        self, hidden_states: torch.Tensor, device: int,
    ) -> torch.Tensor:
        input_hidden_states = []
        for i in range(hidden_states.shape[0]):
            inner_hidden = hidden_states[i]
            layer_matrix = inner_hidden / torch.norm(inner_hidden)
            # print("normmmmmmmmmmmmm", torch.norm(inner_hidden))
            #  expectd shape (seqlen * hidden_state dim)
            normed = layer_matrix - torch.mean(layer_matrix)
            # hidden_state * hidden_state
            # m_exp = torch.mean(normed, dim=0)
            # x = normed - m_exp[None,:]
            # covariance = 1 / (x.size(0) - 1) * x.t().mm(x)
            # covariance = covariance.to(torch.float64)
            # # print("covariance", covariance)
            # eig_vals, eig_vecs = torch.eig(covariance, True)
            # # print("eig_vals", eig_vals.shape)
            # sort_perm = torch.argsort(eig_vals[:,:-1], dim=0, descending=True)
            # sort_perm = sort_perm.view(-1,)
            # eig_vals, indices = torch.sort(eig_vals[:,:-1],0,True)
            # eig_vecs = eig_vecs[:,indices]
            # eig_vecs = torch.squeeze(eig_vecs, 2)
            # print("eig_vals", eig_vals)
            # print("eig_shape", eig_vecs.shape)
            # m1 = torch.transpose(eig_vecs,0,1)
            # print("m1",m1)
            #from here
            covariance = np.cov(normed.cpu().detach().numpy(), rowvar=False)
            #  eig_vals = 1 * hidden_state , eig_vecs = hidden_state * hidden_state
            eig_vals, eig_vecs = np.linalg.eigh(covariance)
            sort_perm = eig_vals[::-1].argsort()
            eig_vals[::-1].sort()
            eig_vecs = eig_vecs[:, sort_perm]
            #  m1 shape is (hidden_state * hidden_state)
            m1 = np.column_stack((eig_vecs))
            m1 = torch.from_numpy(m1)
            # print("m1_after",m1)
            # to there
            #  m2 shape is (hidden_state ,1)
            m2 = torch.zeros(hidden_states.shape[-1], 1)
            # according to the paper alpha should only be draw once per augmentation (not once per channel)
            # from here2
            alpha = np.random.normal(0, 0.1)
            m2[:,0] = alpha * torch.from_numpy(eig_vals)
            # alpha = torch.normal(0,0.01, size=(1,1)).to(device=device)
            # print("alpha shape", alpha.shape)
            # print("eig_vals", eig_vals.shape)
            # m2 = alpha * eig_vals
            # print("m2", m2)
            # add_vect = np.matrix(m1) * np.matrix(m2)
            add_vect = torch.matmul(m1.float(), m2)
            layer_matrix += torch.sum(add_vect)
            # print("layermatrix fixed", torch.sum(add_vect))
            # add_vect2 = torch.matmul(m3.float(), m2)
            # layer_matrix += torch.sum(add_vect2)
            # print("layermatrix after fixed", torch.sum(add_vect))

            layer_matrix = layer_matrix * torch.norm(inner_hidden)

            # print("shape of layer matrix is $$$$$$$$$", layer_matrix.shape)
            input_hidden_states.append(layer_matrix)
        input_hidden_states = torch.stack(input_hidden_states, dim=0)
        return input_hidden_states