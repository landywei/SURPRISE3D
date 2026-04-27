import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gorilla
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.dist_utils import download_cached_file
from lavis.models.reason3d_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.reason3d_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.reason3d_models.mask_decoder import MaskDecoder
from lavis.models.reason3d_models.point_extractor import PointExtractor
from lavis.models.reason3d_models.seg_loss import Criterion
from lavis.common.utils import is_url


def _is_missing_cfg_val(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if not s:
        return True
    return s.lower() in ("null", "none", "~")


def _normalize_checkpoint_path(url_or_filename):
    if url_or_filename is None:
        return None
    p = str(url_or_filename).strip().strip('"').strip("'")
    p = os.path.expanduser(p)
    if not is_url(p) and not os.path.isabs(p):
        p = os.path.abspath(p)
    return p


@registry.register_model("reason3d_t5")
class Reason3DT5(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        apply_lemmatizer=False,
        point_encoder_cfg=None,
        mask_decoder_cfg=None,
        seg_criterion_cfg=None,
        pred_confidence=0.5
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_seg() result with lemmas.
        """
        super().__init__()

        self.encoder = PointExtractor(**point_encoder_cfg)
        self.mask_decoder = MaskDecoder(**mask_decoder_cfg)
        gorilla.load_checkpoint(self.encoder, point_encoder_cfg["pretrained"], strict=False, map_location='cpu')
        
        self.pc_adapter = nn.Linear(point_encoder_cfg["media"], 1408)
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        num_added_tokens = self.t5_tokenizer.add_tokens("[SEG]")
        self.seg_token_idx = self.t5_tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        in_dim = self.t5_model.config.hidden_size
        out_dim = mask_decoder_cfg["d_text"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self.criterion = Criterion(**seg_criterion_cfg)
        self.pred_confidence = pred_confidence

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load full Reason3D weights. Prefer ``model.reason3d_checkpoint`` so it is not confused with
        ``point_encoder_cfg.pretrained`` (SPFormer) or the BLIP2 default ``model.pretrained`` URL from
        the merged base YAML. Falls back to ``model.pretrained`` when it points to a real file/URL.
        """
        if cfg.get("load_finetuned", False):
            return super().load_checkpoint_from_config(cfg, **kwargs)
        raw = cfg.get("reason3d_checkpoint", None)
        if _is_missing_cfg_val(raw):
            raw = cfg.get("pretrained", None)
        if _is_missing_cfg_val(raw):
            raise RuntimeError(
                "No Reason3D checkpoint configured. Pass e.g. "
                "`--options model.reason3d_checkpoint=/absolute/path/to/reason3d.pth` "
                "(recommended; see scripts/run_surprise_zeroshot_eval.sh), or set `model.pretrained` "
                "to that .pth for backward compatibility. Relative paths are resolved from cwd."
            )
        self.load_from_pretrained(raw)

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
        
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768 #2, 32, 768
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
                output_hidden_states=True
            )
            seq_out = outputs["decoder_hidden_states"][-1]
            seg_token_index = targets == self.seg_token_idx
        
            seq_out = seq_out[seg_token_index]

            text_features = self.text_hidden_fcs[0](seq_out).unsqueeze(1)
            samples["text_features"] = text_features
            out = self.mask_decoder(**samples)
            seg_loss, log_vars = self.criterion(out, samples["gt_pmasks"], samples["gt_spmasks"], None)

            loss = outputs.loss
            loss = loss + seg_loss
            return {"loss": loss}

    def _seg_out_from_t5_greedy_then_teacher(
        self,
        inputs_embeds: torch.Tensor,
        encoder_atts: torch.Tensor,
        num_beams: int,
        max_len: int,
        min_len: int,
        length_penalty: float,
        repetition_penalty: float,
        output_device: torch.device,
        no_repeat_ngram_size: int = 0,
    ):
        """
        Greedy decode (num_beams=1 in our callers) without per-step ``output_hidden_states`` in ``generate()``
        (huge memory), then one teacher-forced forward to read ``decoder_hidden_states`` like in ``forward()``
        (same ``labels``-based [SEG] indexing as training; numerics match greedy unrolling).
        Encoder runs once; ``generate`` and the second call reuse ``encoder_outputs``.
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
            num_beams=num_beams,
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
        # Same indexing as training ``forward`` (``targets == seg_token`` on label ids).
        seq_h = t5_out.decoder_hidden_states[-1]
        seg_index = gen_seq == self.seg_token_idx
        picked = seq_h[seg_index]
        hdim = int(self.t5_model.config.hidden_size)
        if picked.numel() == 0:
            return torch.zeros((1, hdim), device=output_device, dtype=seq_h.dtype), gen_seq[0].detach()
        return picked.float().mean(dim=0, keepdim=True), gen_seq[0].detach()

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
            result = self.mask_decoder(**samples)
            result["decoded_text"] = self.t5_tokenizer.decode(
                gen_ids.tolist(), skip_special_tokens=True
            )

            return result

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

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

        model = cls(
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            apply_lemmatizer=apply_lemmatizer,
            point_encoder_cfg=point_encoder_cfg,
            mask_decoder_cfg=mask_decoder_cfg,
            seg_criterion_cfg=seg_criterion_cfg,
            pred_confidence=pred_confidence
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        bert_id = "bert-base-uncased"
        encoder_config = BertConfig.from_pretrained(bert_id)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(bert_id, config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        path = _normalize_checkpoint_path(url_or_filename)
        if path is None:
            raise RuntimeError("checkpoint path is empty")
        if is_url(path):
            cached_file = download_cached_file(path, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
        else:
            parent = os.path.dirname(path) or "."
            hint = ""
            if os.path.isdir(parent):
                hint = (
                    f" Directory {parent!r} exists, but there is no file {os.path.basename(path)!r}."
                )
            raise RuntimeError(
                "checkpoint url or path is invalid: "
                f"raw={url_or_filename!r} -> resolved={path!r} cwd={os.getcwd()}.{hint}"
            )

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % path)

        return msg