"""
Reason3D + geometry-aware superpoint refinement (separate from baseline ``reason3d_t5``).
"""

from __future__ import annotations

import logging

import torch

from lavis.common.registry import registry
from lavis.models.reason3d_models.reason3d_t5 import Reason3DT5
from lavis.models.reason3d_geo.geo_relational import GeoRelationalModule


@registry.register_model("reason3d_t5_geo")
class Reason3DT5Geo(Reason3DT5):
    """
    Same as ``Reason3DT5`` but refines ``sp_feats`` with ``GeoRelationalModule`` after
    ``[SEG]`` text features are formed and before ``MaskDecoder``.

    Requires ``coords_float`` in the batch (use dataset builder ``3d_refer_geo``).
    """

    def __init__(self, geo_relational_cfg=None, **kwargs):
        point_encoder_cfg = kwargs["point_encoder_cfg"]
        mask_decoder_cfg = kwargs["mask_decoder_cfg"]
        super().__init__(**kwargs)
        gcfg = dict(geo_relational_cfg or {})
        knn_k = int(gcfg.pop("knn_k", 16))
        knn_chunk_size = int(gcfg.pop("knn_chunk_size", 512))
        use_checkpoint = bool(gcfg.pop("use_checkpoint", True))
        num_layers = int(gcfg.pop("num_layers", 2))
        hidden_dim = int(gcfg.pop("hidden_dim", 128))
        dropout = float(gcfg.pop("dropout", 0.0))
        if gcfg:
            logging.warning("Reason3DT5Geo: ignoring unknown geo_relational_cfg keys: %s", sorted(gcfg.keys()))
        self.geo_relational = GeoRelationalModule(
            in_dim=int(point_encoder_cfg["media"]),
            cond_dim=int(mask_decoder_cfg["d_text"]),
            knn_k=knn_k,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            knn_chunk_size=knn_chunk_size,
            use_checkpoint=use_checkpoint,
        )

    def _maybe_apply_geo_relational(self, samples, text_features):
        if samples.get("coords_float") is None:
            return
        cond = text_features.squeeze(1)
        samples["sp_feats"] = self.geo_relational(
            samples["sp_feats"],
            samples["coords_float"],
            samples["superpoints"],
            samples["batch_offsets"],
            cond,
        )

    def forward(self, samples):
        with self.maybe_autocast():
            answer = samples["answer"]
            text_input = samples["text_input"]
            n_answers = samples["n_answers"]
            sp_feats = self.encoder(samples)
            samples["sp_feats"] = sp_feats
            x_feat, batch_mask = self.mask_decoder.get_batches(sp_feats, samples["batch_offsets"])
            pc_embeds = x_feat
            pc_embeds = self.pc_adapter(pc_embeds)
            image_atts = (~batch_mask).long()

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if self.prompt:
            text_input = [self.prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=400,
                return_tensors="pt",
            ).to(pc_embeds.device)
            output_tokens = self.t5_tokenizer(
                answer,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(pc_embeds.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, n in enumerate(n_answers):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
            )
            seq_out = outputs["decoder_hidden_states"][-1]
            seg_token_index = targets == self.seg_token_idx

            seq_out = seq_out[seg_token_index]

            text_features = self.text_hidden_fcs[0](seq_out).unsqueeze(1)
            samples["text_features"] = text_features
            self._maybe_apply_geo_relational(samples, text_features)
            out = self.mask_decoder(**samples)
            seg_loss, log_vars = self.criterion(out, samples["gt_pmasks"], samples["gt_spmasks"], None)

            loss = outputs.loss
            loss = loss + seg_loss
            return {"loss": loss}

    def predict_seg(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs,
    ):
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            text_input = samples["text_input"]
            sp_feats = self.encoder(samples)
            samples["sp_feats"] = sp_feats
            pc_embeds, batch_mask = self.mask_decoder.get_batches(sp_feats, samples["batch_offsets"])
            pc_embeds = self.pc_adapter(pc_embeds)
            image_atts = (~batch_mask).long()

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(text_input, str):
            text_input = [text_input]

        prompt = self.prompt

        if prompt:
            text_input = [prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            seg_out, gen_ids = self._seg_out_from_t5_greedy_then_teacher(
                inputs_embeds=inputs_embeds,
                encoder_atts=encoder_atts,
                num_beams=num_beams,
                max_len=max_len,
                min_len=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                output_device=pc_embeds.device,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            text_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
            samples["text_features"] = text_features
            self._maybe_apply_geo_relational(samples, text_features)
            result = self.mask_decoder(**samples)
            result["decoded_text"] = self.t5_tokenizer.decode(
                gen_ids.tolist(), skip_special_tokens=True
            )

            return result

    @classmethod
    def from_config(cls, cfg):
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        mask_decoder_cfg = cfg.get("mask_decoder_cfg")
        seg_criterion_cfg = cfg.get("seg_criterion_cfg")
        pred_confidence = cfg.get("pred_confidence")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        prompt = cfg.get("prompt", "")
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        geo_relational_cfg = cfg.get("geo_relational_cfg", None)

        model = cls(
            geo_relational_cfg=geo_relational_cfg,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            apply_lemmatizer=apply_lemmatizer,
            point_encoder_cfg=point_encoder_cfg,
            mask_decoder_cfg=mask_decoder_cfg,
            seg_criterion_cfg=seg_criterion_cfg,
            pred_confidence=pred_confidence,
        )
        model.load_checkpoint_from_config(cfg)

        return model
