"""
Reason3D + chain v3 CoT (architectural extension over chain v3 loss-only).

Adds, on top of ``Reason3DT5ChainV3``:

1. A learnable ``MaskPoolToken`` projection
   :math:`\\mathbf{W}: \\mathbb{R}^{d_{sp}} \\to \\mathbb{R}^{d_{T5}}` that turns
   a mass-pooled superpoint feature into a single T5-aligned token.
2. A two-pass training forward for samples whose target contains two
   ``[SEG]`` tokens (CoT samples):
     - Pass 1 uses the standard encoder memory (Q-Former + text); we
       extract the hidden state at the **first** ``[SEG]``, decode the
       landmark mask :math:`M_1`, and compute
       :math:`\\mathbf{t}_{\\mathrm{pool}} = W\\,\\mathrm{pool}(M_1.\\mathrm{detach}(),\\, f_p)`.
     - Pass 2 re-runs T5 with encoder memory augmented by
       :math:`\\mathbf{t}_{\\mathrm{pool}}` and extracts the hidden state at
       the **second** ``[SEG]``; this is the final segmentation query.
3. A per-sample LM-loss split: pass-1 LM loss is masked out for CoT
   samples (they get pass-2 LM loss instead); non-CoT samples get
   pass-1 LM loss as in chain v3.

W1-pure: there is **no auxiliary loss term on** :math:`M_1`. Stop-gradient
on :math:`\\mathbf{t}_{\\mathrm{pool}}` (via ``M_1.detach()``) prevents pass-2
seg loss from reshaping :math:`M_1` away from being a recognizable class
mask under the pretrained class-segmentation prior.

Inference (``predict_seg``) is a pause-resume two-pass decode:

- Pass 1: greedy decode the full response from (Q-Former + text);
  extract the first ``[SEG]``, decode :math:`M_1`, compute
  :math:`\\mathbf{t}_{\\mathrm{pool}}`.
- Pass 2: re-encode with :math:`\\mathbf{t}_{\\mathrm{pool}}` appended to the
  encoder's ``inputs_embeds``; greedy decode again, extract the
  **last** ``[SEG]``, decode :math:`M_2` -- the final segmentation.

For samples where pass-1 emits only one ``[SEG]`` (e.g. on non-relational
queries during training-free eval), the model degenerates to chain v3
single-pass behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

from lavis.common.registry import registry
from lavis.models.reason3d_models.reason3d_t5_chainv3 import Reason3DT5ChainV3


def _seg_positions_per_sample(
    targets: torch.Tensor,
    seg_token_idx: int,
) -> List[List[int]]:
    """For each row of ``targets``, return the list of column indices where
    the value equals ``seg_token_idx``.

    ``targets`` may use ``-100`` as ignore; those positions cannot match
    a positive token id so they are naturally excluded.
    """
    out: List[List[int]] = []
    seg_mask = (targets == seg_token_idx)
    for b in range(targets.shape[0]):
        idxs = torch.nonzero(seg_mask[b], as_tuple=False).flatten().tolist()
        out.append([int(i) for i in idxs])
    return out


def _mass_pool_per_sample(
    sp_feats: torch.Tensor,        # [N_total, d_sp]
    batch_offsets: torch.Tensor,   # [B+1]
    masks_sp: torch.Tensor,        # [B_sub, M_max] (post-batch-mask, padded)
    sub_to_full: List[int],        # length B_sub; index of each sub sample in the full batch
) -> torch.Tensor:
    """Compute :math:`\\mathrm{pool}_b = \\sum_p \\sigma(M_b[p])\\, f_p / \\sum_p \\sigma(M_b[p])`
    for each sample ``b`` in the sub-batch.

    Returns a tensor ``[B_sub, d_sp]``. Detaches mask weights (W1-pure).
    """
    pooled: List[torch.Tensor] = []
    for k, b in enumerate(sub_to_full):
        start = int(batch_offsets[b].item())
        end = int(batch_offsets[b + 1].item())
        n_sp_b = end - start
        if n_sp_b <= 0:
            pooled.append(torch.zeros(sp_feats.shape[1], device=sp_feats.device, dtype=sp_feats.dtype))
            continue
        feats_b = sp_feats[start:end]  # [n_sp_b, d_sp]
        # ``masks_sp`` is padded along the M_max axis; only the first n_sp_b
        # superpoints are valid for sample b.
        logits_b = masks_sp[k, :n_sp_b]
        weights_b = logits_b.detach().float().sigmoid()
        denom = weights_b.sum().clamp_min(1e-6)
        pool = (weights_b.unsqueeze(-1) * feats_b.float()).sum(dim=0) / denom
        pooled.append(pool.to(sp_feats.dtype))
    return torch.stack(pooled, dim=0)  # [B_sub, d_sp]


@registry.register_model("reason3d_t5_chainv3_cot")
class Reason3DT5ChainV3CoT(Reason3DT5ChainV3):
    """Chain v3 with multi-step CoT mass-pool feedback (W1-pure).

    Inherits ``CriterionV3`` plumbing from ``Reason3DT5ChainV3``. Adds a
    single linear projection ``mask_pool_proj`` on top of the parent.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        d_sp = int(kwargs["point_encoder_cfg"]["media"])
        d_t5 = int(self.t5_model.config.hidden_size)
        self.mask_pool_proj = nn.Linear(d_sp, d_t5)
        nn.init.xavier_uniform_(self.mask_pool_proj.weight)
        nn.init.zeros_(self.mask_pool_proj.bias)
        self.mask_pool_proj.requires_grad_(True)

    # =================================================================
    # Forward (training)
    # =================================================================

    def forward(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Stash chain v3 extras for ``CriterionV3`` (parent contract).
        self.criterion._pending_extras = {
            "gt_pmasks_per_instance": samples.get("gt_pmasks_per_instance"),
            "gt_spmasks_per_instance": samples.get("gt_spmasks_per_instance"),
            "superpoints": samples.get("superpoints"),
            "batch_offsets": samples.get("batch_offsets"),
        }
        try:
            return self._forward_cot(samples)
        finally:
            self.criterion._pending_extras = None

    def _forward_cot(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with self.maybe_autocast():
            answer = samples["answer"]
            text_input = samples["text_input"]
            n_answers = samples["n_answers"]
            sp_feats = self.encoder(samples)
            samples["sp_feats"] = sp_feats
            x_feat, batch_mask = self.mask_decoder.get_batches(sp_feats, samples["batch_offsets"])
            pc_embeds = self.pc_adapter(x_feat)
            image_atts = (~batch_mask).long()

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)         # [B, n_q, d_t5]
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=pc_embeds.device)

        if self.prompt:
            text_input = [self.prompt.format(q) for q in text_input]

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input, padding="longest", truncation=True,
                max_length=400, return_tensors="pt",
            ).to(pc_embeds.device)
            output_tokens = self.t5_tokenizer(
                answer, padding="longest", truncation=True,
                max_length=300, return_tensors="pt",
            ).to(pc_embeds.device)

            # Duplicate per-sample fields by ``n_answers`` (kept for parity with
            # base; chain v3 CoT typically emits exactly one answer per sample).
            batch_inputs_t5: List[torch.Tensor] = []
            batch_atts_t5: List[torch.Tensor] = []
            batch_input_tokens_input_ids: List[torch.Tensor] = []
            batch_input_tokens_atts: List[torch.Tensor] = []
            row_to_scene: List[int] = []
            for b, n in enumerate(n_answers.tolist() if torch.is_tensor(n_answers) else list(n_answers)):
                for _ in range(int(n)):
                    batch_inputs_t5.append(inputs_t5[b])
                    batch_atts_t5.append(atts_t5[b])
                    batch_input_tokens_input_ids.append(input_tokens.input_ids[b])
                    batch_input_tokens_atts.append(input_tokens.attention_mask[b])
                    row_to_scene.append(b)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)

            text_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds_p1 = torch.cat([batch_inputs_t5, text_embeds], dim=1)
            encoder_atts_p1 = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            decoder_atts = output_tokens.attention_mask
            B_eff = targets.shape[0]

            # Identify CoT vs non-CoT rows by counting [SEG] tokens in target ids.
            seg_pos = _seg_positions_per_sample(targets, self.seg_token_idx)
            cot_rows: List[int] = [b for b, ps in enumerate(seg_pos) if len(ps) >= 2]
            non_cot_rows: List[int] = [b for b, ps in enumerate(seg_pos) if len(ps) == 1]
            # Rows with zero [SEG]s are degenerate; treat them as non-CoT for
            # LM-loss accounting and skip the seg path -- the dataset should
            # not produce these but we guard anyway.
            zero_seg_rows: List[int] = [b for b, ps in enumerate(seg_pos) if len(ps) == 0]

            # ---- Pass 1: standard forward over the full effective batch ----
            # Mask out CoT samples' targets so pass-1 LM loss only contributes
            # for non-CoT (and zero-seg) samples.
            targets_p1 = targets.clone()
            if cot_rows:
                cot_idx = torch.tensor(cot_rows, device=targets.device, dtype=torch.long)
                targets_p1[cot_idx, :] = -100
            n_p1_valid = int((targets_p1 != -100).sum().item())

            outputs_p1 = self.t5_model(
                inputs_embeds=inputs_embeds_p1,
                attention_mask=encoder_atts_p1,
                decoder_attention_mask=decoder_atts,
                return_dict=True,
                labels=targets_p1,
                output_hidden_states=True,
            )
            seq_out_p1 = outputs_p1["decoder_hidden_states"][-1]   # [B_eff, T_dec, d_t5]
            lm_loss_p1 = outputs_p1.loss if n_p1_valid > 0 else seq_out_p1.new_zeros(())

            # ---- Pass 2: CoT-only forward with mass-pool token appended ----
            lm_loss_p2 = seq_out_p1.new_zeros(())
            n_p2_valid = 0
            if cot_rows:
                # h_seg_1: hidden state at the FIRST [SEG] of each CoT row.
                h_seg_1_list: List[torch.Tensor] = []
                for b in cot_rows:
                    p1 = int(seg_pos[b][0])
                    h_seg_1_list.append(seq_out_p1[b, p1])
                h_seg_1 = torch.stack(h_seg_1_list, dim=0)         # [B_cot, d_t5]

                # Decode M_1 over the CoT subset. Run mask decoder under no-grad:
                # there is no loss term on M_1 (W1-pure) and we will detach it
                # before pooling, so its parameters are not updated through this
                # path (they are still updated through the M_2 path below).
                # Map cot rows back to scene indices for batch_offsets / sp_feats.
                cot_scenes: List[int] = [row_to_scene[b] for b in cot_rows]
                sub_batch_offsets = self._sub_batch_offsets(samples["batch_offsets"], cot_scenes, sp_feats.device)
                sub_sp_feats = self._sub_sp_feats(sp_feats, samples["batch_offsets"], cot_scenes)
                with torch.no_grad():
                    text_features_p1 = self.text_hidden_fcs[0](h_seg_1).unsqueeze(1)  # [B_cot, 1, d_text]
                    out_M1 = self.mask_decoder(
                        sp_feats=sub_sp_feats,
                        batch_offsets=sub_batch_offsets,
                        text_features=text_features_p1,
                    )
                # ``out_M1["masks"]`` has shape [B_cot, n_q, M_max]; n_q = 1.
                M1_masks = out_M1["masks"].squeeze(1)                 # [B_cot, M_max]

                # Mass-pool to get a single d_sp token per CoT sample, then
                # project to d_t5.
                pool_unproj = _mass_pool_per_sample(
                    sp_feats=sp_feats,
                    batch_offsets=samples["batch_offsets"],
                    masks_sp=M1_masks,
                    sub_to_full=cot_scenes,
                )                                                     # [B_cot, d_sp]
                t_pool = self.mask_pool_proj(pool_unproj)             # [B_cot, d_t5]

                # Pass 2 inputs: (Q-Former + text + t_pool) for CoT rows only.
                cot_idx_t = torch.tensor(cot_rows, device=targets.device, dtype=torch.long)
                inputs_embeds_p2 = torch.cat(
                    [
                        inputs_embeds_p1.index_select(0, cot_idx_t),
                        t_pool.unsqueeze(1),
                    ],
                    dim=1,
                )
                encoder_atts_p2 = torch.cat(
                    [
                        encoder_atts_p1.index_select(0, cot_idx_t),
                        torch.ones(len(cot_rows), 1, dtype=encoder_atts_p1.dtype, device=encoder_atts_p1.device),
                    ],
                    dim=1,
                )
                targets_p2 = targets.index_select(0, cot_idx_t)
                decoder_atts_p2 = decoder_atts.index_select(0, cot_idx_t)
                n_p2_valid = int((targets_p2 != -100).sum().item())

                outputs_p2 = self.t5_model(
                    inputs_embeds=inputs_embeds_p2,
                    attention_mask=encoder_atts_p2,
                    decoder_attention_mask=decoder_atts_p2,
                    return_dict=True,
                    labels=targets_p2,
                    output_hidden_states=True,
                )
                seq_out_p2 = outputs_p2["decoder_hidden_states"][-1]   # [B_cot, T_dec, d_t5]
                lm_loss_p2 = outputs_p2.loss

                # h_seg_2: hidden state at the SECOND (last) [SEG] of each
                # CoT row in pass-2's decoder output.
                h_seg_2_list: List[torch.Tensor] = []
                for k, b in enumerate(cot_rows):
                    p2 = int(seg_pos[b][-1])
                    h_seg_2_list.append(seq_out_p2[k, p2])
                h_seg_2 = torch.stack(h_seg_2_list, dim=0)             # [B_cot, d_t5]
            else:
                h_seg_2 = None

            # ---- Build the final-mask query ``h_seg_final`` per row ----
            # Non-CoT / zero-seg rows: take h_seg from pass 1 at the (only) [SEG]
            # position; zero-seg rows are degenerate and would not normally
            # reach this path -- we feed a zero vector and rely on the dataset
            # to never emit zero-[SEG] targets.
            h_seg_final = seq_out_p1.new_zeros(B_eff, seq_out_p1.shape[-1])
            for b in non_cot_rows:
                p = int(seg_pos[b][0])
                h_seg_final[b] = seq_out_p1[b, p]
            if cot_rows:
                for k, b in enumerate(cot_rows):
                    h_seg_final[b] = h_seg_2[k]
            # zero_seg_rows already left as zeros.

            text_features = self.text_hidden_fcs[0](h_seg_final).unsqueeze(1)  # [B_eff, 1, d_text]
            samples["text_features"] = text_features
            out = self.mask_decoder(**samples)
            seg_loss, _ = self.criterion(out, samples["gt_pmasks"], samples["gt_spmasks"], None)

            # Weighted LM loss: (n_p1*loss_p1 + n_p2*loss_p2) / (n_p1 + n_p2).
            total_n = max(n_p1_valid + n_p2_valid, 1)
            lm_loss = (lm_loss_p1 * n_p1_valid + lm_loss_p2 * n_p2_valid) / total_n

            return {"loss": lm_loss + seg_loss}

    # =================================================================
    # Helpers for sub-batch indexing
    # =================================================================

    @staticmethod
    def _sub_batch_offsets(
        batch_offsets: torch.Tensor,
        scene_indices: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a contiguous ``[len(scene_indices)+1]`` offset tensor for the
        sub-batch of selected scenes, re-indexing superpoints to start at 0.
        """
        offsets: List[int] = [0]
        running = 0
        for b in scene_indices:
            n_sp_b = int(batch_offsets[b + 1].item() - batch_offsets[b].item())
            running += n_sp_b
            offsets.append(running)
        return torch.tensor(offsets, dtype=batch_offsets.dtype, device=device)

    @staticmethod
    def _sub_sp_feats(
        sp_feats: torch.Tensor,
        batch_offsets: torch.Tensor,
        scene_indices: List[int],
    ) -> torch.Tensor:
        """Concatenate ``sp_feats`` slices for the listed scenes."""
        chunks: List[torch.Tensor] = []
        for b in scene_indices:
            start = int(batch_offsets[b].item())
            end = int(batch_offsets[b + 1].item())
            chunks.append(sp_feats[start:end])
        return torch.cat(chunks, dim=0)

    # =================================================================
    # Inference (pause-resume two-pass decode)
    # =================================================================

    def _greedy_then_teacher(
        self,
        inputs_embeds: torch.Tensor,
        encoder_atts: torch.Tensor,
        max_len: int,
        min_len: int,
        length_penalty: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decode then teacher-forced forward to read decoder hidden
        states; encoder runs once and is reused for both calls.

        Returns:
            gen_seq: ``[B, T_gen]`` long tensor of generated token ids.
            decoder_hidden: ``[B, T_gen, d_t5]`` last-layer decoder states
                aligned 1:1 with ``gen_seq`` (same indexing as training).
        """
        enc_out = self.t5_model.get_encoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            return_dict=True,
        )
        gen_kwargs = dict(
            encoder_outputs=enc_out,
            attention_mask=encoder_atts,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
        if no_repeat_ngram_size and int(no_repeat_ngram_size) > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        gen = self.t5_model.generate(**gen_kwargs)
        gen_seq = gen.sequences
        with torch.inference_mode():
            t5_out = self.t5_model(
                encoder_outputs=enc_out,
                attention_mask=encoder_atts,
                labels=gen_seq,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
        decoder_hidden = t5_out.decoder_hidden_states[-1]
        return gen_seq, decoder_hidden

    def predict_seg(
        self,
        samples: Dict[str, Any],
        num_beams: int = 5,
        inference_method: str = "generate",
        max_len: int = 200,
        min_len: int = 1,
        num_ans_candidates: int = 128,
        answer_list: Optional[List[str]] = None,
        prompt: str = "",
        length_penalty: float = -1,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pause-resume two-pass inference.

        Pass 1: greedy decode from ``(Q-Former + text)``; if the response
        contains $\\geq 2$ ``[SEG]``s, decode :math:`M_1` from the first
        ``[SEG]`` and compute :math:`\\mathbf{t}_{\\mathrm{pool}}`.

        Pass 2 (only if pass 1 emitted $\\geq 2$ ``[SEG]``s): re-encode
        with :math:`\\mathbf{t}_{\\mathrm{pool}}` appended to ``inputs_embeds``
        and decode again; the final segmentation mask is queried by
        the **last** ``[SEG]`` of pass-2's response.

        Falls back to chain v3 single-pass behavior on any degenerate
        case (zero ``[SEG]``, only one ``[SEG]``, or pass-2 fails to
        emit ``[SEG]``).
        """
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
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=pc_embeds.device)

        if isinstance(text_input, str):
            text_input = [text_input]
        if self.prompt:
            text_input = [self.prompt.format(q) for q in text_input]

        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input, padding="longest", return_tensors="pt",
            ).to(pc_embeds.device)
            text_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds_p1 = torch.cat([inputs_t5, text_embeds], dim=1)
            encoder_atts_p1 = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            # ---- Pass 1: greedy decode + teacher-forced hidden-state read ----
            gen_seq_p1, hidden_p1 = self._greedy_then_teacher(
                inputs_embeds=inputs_embeds_p1,
                encoder_atts=encoder_atts_p1,
                max_len=max_len,
                min_len=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            seg_pos_p1 = _seg_positions_per_sample(gen_seq_p1, self.seg_token_idx)

            # Determine which samples should run pass 2 (>= 2 [SEG]s in pass-1).
            cot_rows: List[int] = [b for b, ps in enumerate(seg_pos_p1) if len(ps) >= 2]

            decoded_text_p1 = [
                self.t5_tokenizer.decode(gen_seq_p1[b].tolist(), skip_special_tokens=True)
                for b in range(gen_seq_p1.shape[0])
            ]

            if not cot_rows:
                # Single-pass fallback: use last [SEG] from pass 1 per sample
                # (or zero vector if none).
                h_seg_final = self._h_seg_per_sample_or_zero(hidden_p1, seg_pos_p1, take="last")
                text_features = self.text_hidden_fcs[0](h_seg_final).unsqueeze(1)
                samples["text_features"] = text_features
                result = self.mask_decoder(**samples)
                result["decoded_text"] = decoded_text_p1[0] if len(decoded_text_p1) == 1 else decoded_text_p1
                result["chainv3_cot"] = {
                    "did_two_pass": False,
                    "n_seg_pass1": [len(ps) for ps in seg_pos_p1],
                    "n_seg_pass2": [0] * gen_seq_p1.shape[0],
                    "decoded_text_pass1": (
                        decoded_text_p1[0] if len(decoded_text_p1) == 1 else decoded_text_p1
                    ),
                }
                # No pass-2 ran, so no intermediate M_1 was decoded; emit None
                # per sample to keep the contract uniform with the two-pass branch.
                result["intermediate_sp_masks"] = [None] * gen_seq_p1.shape[0]
                return result

            # ---- Compute M_1 + t_pool for CoT samples ----
            h_seg_1_list: List[torch.Tensor] = []
            for b in cot_rows:
                p1 = int(seg_pos_p1[b][0])
                h_seg_1_list.append(hidden_p1[b, p1])
            h_seg_1 = torch.stack(h_seg_1_list, dim=0)

            sub_batch_offsets = self._sub_batch_offsets(samples["batch_offsets"], cot_rows, sp_feats.device)
            sub_sp_feats = self._sub_sp_feats(sp_feats, samples["batch_offsets"], cot_rows)
            with torch.no_grad():
                text_features_p1 = self.text_hidden_fcs[0](h_seg_1).unsqueeze(1)
                out_M1 = self.mask_decoder(
                    sp_feats=sub_sp_feats,
                    batch_offsets=sub_batch_offsets,
                    text_features=text_features_p1,
                )
            M1_masks = out_M1["masks"].squeeze(1)
            pool_unproj = _mass_pool_per_sample(
                sp_feats=sp_feats,
                batch_offsets=samples["batch_offsets"],
                masks_sp=M1_masks,
                sub_to_full=cot_rows,
            )
            t_pool = self.mask_pool_proj(pool_unproj)

            # ---- Pass 2: re-encode with t_pool appended for CoT rows ----
            cot_idx_t = torch.tensor(cot_rows, device=hidden_p1.device, dtype=torch.long)
            inputs_embeds_p2 = torch.cat(
                [
                    inputs_embeds_p1.index_select(0, cot_idx_t),
                    t_pool.unsqueeze(1),
                ],
                dim=1,
            )
            encoder_atts_p2 = torch.cat(
                [
                    encoder_atts_p1.index_select(0, cot_idx_t),
                    torch.ones(len(cot_rows), 1, dtype=encoder_atts_p1.dtype, device=encoder_atts_p1.device),
                ],
                dim=1,
            )
            gen_seq_p2, hidden_p2 = self._greedy_then_teacher(
                inputs_embeds=inputs_embeds_p2,
                encoder_atts=encoder_atts_p2,
                max_len=max_len,
                min_len=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            seg_pos_p2 = _seg_positions_per_sample(gen_seq_p2, self.seg_token_idx)

            decoded_text_p2 = [
                self.t5_tokenizer.decode(gen_seq_p2[k].tolist(), skip_special_tokens=True)
                for k in range(gen_seq_p2.shape[0])
            ]

            # ---- Build h_seg_final per sample ----
            B = inputs_embeds_p1.shape[0]
            h_seg_final = hidden_p1.new_zeros(B, hidden_p1.shape[-1])
            n_seg_p2 = [0] * B
            # Non-CoT rows: take last [SEG] from pass 1.
            for b in range(B):
                if b in cot_rows:
                    continue
                ps = seg_pos_p1[b]
                if ps:
                    h_seg_final[b] = hidden_p1[b, int(ps[-1])]
            # CoT rows: take last [SEG] from pass 2; fallback to pass-1's last
            # [SEG] if pass 2 fails to emit one.
            for k, b in enumerate(cot_rows):
                ps2 = seg_pos_p2[k]
                n_seg_p2[b] = len(ps2)
                if ps2:
                    h_seg_final[b] = hidden_p2[k, int(ps2[-1])]
                else:
                    p1_last = int(seg_pos_p1[b][-1])
                    h_seg_final[b] = hidden_p1[b, p1_last]

            text_features = self.text_hidden_fcs[0](h_seg_final).unsqueeze(1)
            samples["text_features"] = text_features
            result = self.mask_decoder(**samples)

            # Attach diagnostics. ``decoded_text`` is the pass-2 text for CoT
            # rows and pass-1 text for non-CoT rows -- the one that produced
            # the final [SEG] used to query the mask.
            decoded_text_final = list(decoded_text_p1)
            for k, b in enumerate(cot_rows):
                decoded_text_final[b] = decoded_text_p2[k]
            result["decoded_text"] = (
                decoded_text_final[0] if len(decoded_text_final) == 1 else decoded_text_final
            )
            result["chainv3_cot"] = {
                "did_two_pass": True,
                "n_seg_pass1": [len(ps) for ps in seg_pos_p1],
                "n_seg_pass2": n_seg_p2,
                "decoded_text_pass1": decoded_text_p1[0] if len(decoded_text_p1) == 1 else decoded_text_p1,
            }
            # Per-sample intermediate (M_1) superpoint logits, sliced to the true
            # n_sp_b for each scene (M1_masks is padded to M_max across the cot
            # sub-batch). None for non-CoT rows. Detached so downstream callers
            # can move/save without holding the autograd graph.
            B = inputs_embeds_p1.shape[0]
            intermediate_sp_masks: List[Optional[torch.Tensor]] = [None] * B
            for k, b in enumerate(cot_rows):
                start = int(samples["batch_offsets"][b].item())
                end = int(samples["batch_offsets"][b + 1].item())
                n_sp_b = end - start
                intermediate_sp_masks[b] = M1_masks[k, :n_sp_b].detach()
            result["intermediate_sp_masks"] = intermediate_sp_masks
            return result

    @staticmethod
    def _h_seg_per_sample_or_zero(
        hidden: torch.Tensor,
        seg_positions: List[List[int]],
        take: str = "last",
    ) -> torch.Tensor:
        """Pick one ``[SEG]`` hidden per row; zero vector if no ``[SEG]``."""
        B, _, d = hidden.shape
        out = hidden.new_zeros(B, d)
        for b, ps in enumerate(seg_positions):
            if not ps:
                continue
            idx = ps[-1] if take == "last" else ps[0]
            out[b] = hidden[b, int(idx)]
        return out
