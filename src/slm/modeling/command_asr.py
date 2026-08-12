"""CommandASR — our speech/text -> tool-call model, in the NATIVE Qwen3-ASR architecture.

Instead of a bespoke fusion, the model IS a `Qwen3ASRForConditionalGeneration`
(vendored official code). The preferred initialization preserves the complete
co-trained Qwen3-ASR-0.6B checkpoint; a stock-Qwen3-0.6B decoder transplant is
retained as an ablation. It is HF-native (save_pretrained/from_pretrained/
push_to_hub) and vLLM/SGLang-servable exactly like Qwen3-ASR. Speech->command
is ASR with a `<tool_call>` target instead of a transcript.

Training trains `model.thinker` (the wrapper has no loss path): its forward fuses audio at the
<|audio_pad|> placeholder (masked_scatter) and returns CE loss on `labels`.
"""
from __future__ import annotations
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.modeling.build import META_ARCH_REGISTRY
from slm.modeling.audio_encoder.qwen3asr_vendor.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "audio_encoder", "qwen3asr_vendor")
_VENDOR_FILES = ["configuration_qwen3_asr.py", "modeling_qwen3_asr.py", "processing_qwen3_asr.py"]
_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_asr.Qwen3ASRConfig",
    "AutoModel": "modeling_qwen3_asr.Qwen3ASRForConditionalGeneration",
    "AutoModelForCausalLM": "modeling_qwen3_asr.Qwen3ASRForConditionalGeneration",
    "AutoProcessor": "processing_qwen3_asr.Qwen3ASRProcessor",
}
_AUDIO_TOKENS = {
    "audio_token": "<|audio_pad|>",
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
}


def _set_audio_token_attrs(tokenizer) -> None:
    """Set and persist the tokenizer attributes required by Qwen3-ASR processors."""
    for name, value in _AUDIO_TOKENS.items():
        setattr(tokenizer, name, value)
        # PreTrainedTokenizerBase.save_pretrained serializes init_kwargs into
        # tokenizer_config.json. Plain instance attributes are not sufficient
        # for standalone vLLM/AutoProcessor reloads.
        tokenizer.init_kwargs[name] = value


def transplant_decoder(asr_model, qwen3_id: str) -> None:
    """Load stock Qwen3-0.6B decoder+lm_head+embed onto asr_model.thinker (keys are identical)."""
    from transformers import AutoModelForCausalLM
    tgt = next(asr_model.thinker.parameters())
    src = AutoModelForCausalLM.from_pretrained(qwen3_id, dtype=tgt.dtype)
    sd = {k: v.to(device=tgt.device, dtype=tgt.dtype) for k, v in src.state_dict().items()}
    if "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    missing, unexpected = asr_model.thinker.load_state_dict(sd, strict=False)
    asr_model.thinker.tie_weights()
    bad = [m for m in missing if not m.startswith("audio_tower.")]
    assert not bad, f"non-audio keys missing after transplant: {bad[:6]}"
    assert not unexpected, f"unexpected keys: {unexpected[:6]}"
    del src


def build_tokenizer(qwen3_id: str, asr_id: str):
    """Qwen3-0.6B tokenizer (keeps its tools+thinking chat template) + the ASR special tokens so
    <|audio_pad|> lands at 151676 (== config.audio_token_id)."""
    from transformers import AutoTokenizer, AddedToken
    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained(qwen3_id)
    atd = json.load(open(hf_hub_download(asr_id, "tokenizer_config.json")))["added_tokens_decoder"]
    have = set(tok.get_vocab())
    add = [AddedToken(atd[i]["content"], special=bool(atd[i].get("special", True)), normalized=False)
           for i in sorted(atd, key=int) if atd[i]["content"] not in have]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    # Qwen3ASRProcessor reads these tokenizer attrs (the real ASR tokenizer sets them).
    _set_audio_token_attrs(tok)
    return tok


def verify_asr_tokenizer_alignment(tokenizer, asr_id: str, model) -> None:
    """Prove that the tool-aware tokenizer is safe for untouched ASR weights.

    We borrow the stock Qwen3 chat template, but the embedding rows remain the
    ASR checkpoint's rows.  Every token already known to the ASR tokenizer must
    therefore keep exactly the same integer ID, and the final vocabulary size
    must match the model embeddings.
    """
    from transformers import AutoTokenizer

    asr_tokenizer = AutoTokenizer.from_pretrained(asr_id, trust_remote_code=True)
    asr_vocab = asr_tokenizer.get_vocab()
    tool_vocab = tokenizer.get_vocab()
    mismatched = [
        (token, token_id, tool_vocab.get(token))
        for token, token_id in asr_vocab.items()
        if tool_vocab.get(token) != token_id
    ]
    if mismatched:
        raise ValueError(
            "stock Qwen3 + ASR tokenizer changes IDs used by the pretrained "
            f"ASR decoder; first mismatches: {mismatched[:5]}"
        )
    embedding_rows = int(model.thinker.model.embed_tokens.num_embeddings)
    max_token_id = max(tool_vocab.values())
    if len(tokenizer) > embedding_rows or max_token_id >= embedding_rows:
        raise ValueError(
            f"tokenizer IDs exceed model embeddings: tokenizer={len(tokenizer)}, "
            f"max_token_id={max_token_id}, embedding_rows={embedding_rows}"
        )
    expected_audio_ids = {
        "<|audio_pad|>": int(model.config.thinker_config.audio_token_id),
        "<|audio_start|>": int(model.config.thinker_config.audio_start_token_id),
        "<|audio_end|>": int(model.config.thinker_config.audio_end_token_id),
    }
    actual_audio_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in expected_audio_ids
    }
    if actual_audio_ids != expected_audio_ids:
        raise ValueError(
            f"audio special-token IDs changed: expected={expected_audio_ids}, "
            f"actual={actual_audio_ids}"
        )


def build_processor(tokenizer, asr_id: str):
    from transformers import WhisperFeatureExtractor
    from slm.modeling.audio_encoder.qwen3asr_vendor.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160,
                                 chunk_length=30, n_fft=400, padding_value=0.0, dither=0.0,
                                 return_attention_mask=True)
    return Qwen3ASRProcessor(feature_extractor=fe, tokenizer=tokenizer, chat_template=tokenizer.chat_template)


def sparse_causal_lm_loss(hidden_states, labels, lm_head):
    """Exact causal CE while projecting only supervised next-token positions.

    Full tool schemas make more than 99% of a speech-to-command sequence prompt tokens with
    label ``-100``. Projecting all of them to the 151k vocabulary needlessly
    dominates memory. This is mathematically identical to standard shifted
    causal-LM loss with ``ignore_index=-100``.
    """
    shift_labels = labels[:, 1:].contiguous()
    active = shift_labels != -100
    if not bool(active.any()):
        raise ValueError("training batch has no supervised target tokens")
    selected_hidden = hidden_states[:, :-1, :][active]
    selected_labels = shift_labels[active]
    logits = lm_head(selected_hidden)
    return F.cross_entropy(logits.float(), selected_labels, reduction="mean")


def assemble(
    asr_id="Qwen/Qwen3-ASR-0.6B",
    qwen3_id="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    keep_asr_audio_embeddings=True,
    attn_implementation="sdpa",
    decoder_init="qwen3",
):
    """Build one native Qwen3-ASR model with an explicit decoder initialization.

    ``decoder_init="qwen3"`` preserves the original hybrid ablation: ASR audio
    tower/projector plus a transplanted stock Qwen3 decoder.

    ``decoder_init="asr"`` keeps the complete co-trained Qwen3-ASR checkpoint
    intact and changes only the tokenizer chat template.  This is the preferred
    production initialization for direct speech-to-command training.
    """
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        asr_id, dtype=dtype, attn_implementation=attn_implementation)
    if decoder_init == "qwen3":
        audio_ids = [model.config.thinker_config.audio_token_id,
                     model.config.thinker_config.audio_start_token_id,
                     model.config.thinker_config.audio_end_token_id]
        saved = None
        if keep_asr_audio_embeddings:
            # preserve the ASR-trained embeddings for the audio delimiter tokens (co-trained with
            # the audio_tower); the rest of the decoder is overwritten by Qwen3-0.6B.
            emb = model.thinker.model.embed_tokens.weight.data
            saved = {t: emb[t].clone() for t in audio_ids}
        transplant_decoder(model, qwen3_id)
        if saved:
            emb = model.thinker.model.embed_tokens.weight.data
            for t, v in saved.items():
                emb[t] = v.to(emb.dtype)
    elif decoder_init != "asr":
        raise ValueError("model.decoder_init must be 'asr' or 'qwen3'")
    tok = build_tokenizer(qwen3_id, asr_id)
    if decoder_init == "asr":
        verify_asr_tokenizer_alignment(tok, asr_id, model)
    proc = build_processor(tok, asr_id)
    model.config.speech_to_action_decoder_init = decoder_init
    # Retain the legacy key so checkpoints and audits from the source STC
    # implementation remain readable during migration.
    model.config.stc_decoder_init = decoder_init
    return model, tok, proc


@META_ARCH_REGISTRY.register()
class CommandASR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        dtype = getattr(torch, mc.get("dtype", "bfloat16"))
        # FC prompts carry the whole 32-tool catalog (~6k tokens); eager attention materializes the
        # full B*H*T^2 score matrix and OOMs even a 0.6B model. SDPA (or flash) avoids that.
        attn = mc.get("attn_implementation", "sdpa")
        if mc.get("from_dir"):
            self.model, self.tokenizer, self.processor = self._load_dir(mc.from_dir, dtype, attn)
        else:
            self.model, self.tokenizer, self.processor = assemble(
                mc.get("asr_pretrained", "Qwen/Qwen3-ASR-0.6B"),
                mc.get("llm_pretrained", "Qwen/Qwen3-0.6B"),
                dtype,
                attn_implementation=attn,
                decoder_init=mc.get("decoder_init", "qwen3"),
            )
        self.audio_pad_id = int(self.model.config.thinker_config.audio_token_id)
        self.sparse_supervised_logits = bool(mc.get("sparse_supervised_logits", False))
        if mc.get("gradient_checkpointing", False):
            # long FC prompts -> recompute activations instead of storing them (use_reentrant=False
            # so it works with frozen embeddings / no input requiring grad)
            self.model.thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.set_stage(mc.get("freeze_stage", "align"))

    @property
    def thinker(self):
        return self.model.thinker

    def set_stage(self, stage: str) -> None:
        """align: train the audio merger (ln_post/proj1/proj2). sft: + text decoder. full: + encoder."""
        for p in self.model.parameters():
            p.requires_grad = False
        merger = ("thinker.audio_tower.ln_post", "thinker.audio_tower.proj1", "thinker.audio_tower.proj2")
        for n, p in self.model.named_parameters():
            if n.startswith(merger):
                p.requires_grad = True
            elif stage in ("sft", "full") and (n.startswith("thinker.model.") or n.startswith("thinker.lm_head")):
                p.requires_grad = True
            elif stage == "full" and n.startswith("thinker.audio_tower"):
                p.requires_grad = True

    def forward(self, batch: dict | None = None, **inputs) -> dict:
        """Run a training batch.

        Accepting both ``model(batch)`` and ``model(**batch)`` keeps the wrapper
        compatible with the local trainer and Hugging Face/Accelerate-style
        callers. Packing-specific FlashAttention kwargs are forwarded unchanged
        to the native Qwen3-ASR thinker.
        """
        if batch is None:
            batch = inputs
        elif inputs:
            raise ValueError("pass either a batch dict or keyword inputs, not both")
        feats = batch.get("input_features")
        if feats is not None:                          # WhisperFeatureExtractor emits float32; cast to model dtype
            feats = feats.to(self.thinker.audio_tower.proj2.weight.dtype)
        flash_kwargs = {
            key: batch[key]
            for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            if key in batch
        }
        if self.sparse_supervised_logits and batch.get("position_ids") is not None:
            # Packed batches already provide reset position IDs and FA2 varlen
            # boundaries, so we can bypass the thinker's unconditional
            # full-vocabulary logits without changing multimodal fusion.
            input_ids = batch["input_ids"]
            inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
            if feats is not None:
                audio_features = self.thinker.get_audio_features(
                    feats,
                    feature_attention_mask=batch.get("feature_attention_mask"),
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                audio_mask = self.thinker.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
            outputs = self.thinker.model(
                attention_mask=batch.get("attention_mask"),
                position_ids=batch["position_ids"],
                inputs_embeds=inputs_embeds,
                use_cache=False,
                **flash_kwargs,
            )
            loss = sparse_causal_lm_loss(
                outputs[0], batch["labels"], self.thinker.lm_head
            )
            return {"loss": loss}
        out = self.thinker(
            input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"),
            input_features=feats, feature_attention_mask=batch.get("feature_attention_mask"),
            position_ids=batch.get("position_ids"), labels=batch.get("labels"),
            use_cache=False, **flash_kwargs)
        # logits ([B,T,vocab]) are unused downstream (the trainer/eval-hook read only .loss; inference
        # uses generate()), so we don't surface them — avoids pinning the large tensor past the step.
        return {"loss": out.loss}

    # ---- standalone HF save / load ----
    def save_pretrained(self, save_dir: str) -> str:
        from slm.modeling.qwen3_tool_calls import assistant_eos_token_id

        os.makedirs(save_dir, exist_ok=True)
        gc = self.model.generation_config           # ASR ships a greedy config with stray sampling params
        im_end = assistant_eos_token_id(self.tokenizer)
        end_of_text = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # vLLM loads generation_config.json from the model path by default.
        # Preserve Qwen3-ASR's two safe terminators and its non-EOS padding ID.
        gc.eos_token_id = [int(end_of_text), im_end]
        gc.pad_token_id = int(end_of_text)
        gc.do_sample = False
        if not getattr(gc, "do_sample", False):
            gc.temperature = gc.top_p = gc.top_k = None
        self.model.save_pretrained(save_dir, safe_serialization=True)
        self.processor.save_pretrained(save_dir)
        for f in _VENDOR_FILES:                       # ship modeling for trust_remote_code reload
            shutil.copy(os.path.join(_VENDOR_DIR, f), os.path.join(save_dir, f))
        cfg_path = os.path.join(save_dir, "config.json")
        cfg = json.load(open(cfg_path))
        cfg["auto_map"] = _AUTO_MAP                   # makes AutoModel.from_pretrained(trust_remote_code) work
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        return save_dir

    @staticmethod
    def _load_dir(save_dir, dtype, attn_implementation="sdpa"):
        """Load a previously-saved merged dir directly (NO re-assembly / transplant)."""
        from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
        from slm.modeling.audio_encoder.qwen3asr_vendor.processing_qwen3_asr import (
            Qwen3ASRProcessor,
        )
        model = AutoModel.from_pretrained(save_dir, trust_remote_code=True, dtype=dtype,
                                          attn_implementation=attn_implementation)
        # Transformers 4.57 can mis-detect locally-saved non-Mistral tokenizers
        # as affected by the Mistral regex migration.  Qwen's regex is already
        # correct; passing False explicitly suppresses the false-positive
        # warning without rewriting the tokenizer pre-tokenizer.
        tok = AutoTokenizer.from_pretrained(save_dir, fix_mistral_regex=False)
        # the audio_token attrs are instance attrs (not persisted) -> restore them
        _set_audio_token_attrs(tok)
        fe = WhisperFeatureExtractor.from_pretrained(save_dir)
        proc = Qwen3ASRProcessor(feature_extractor=fe, tokenizer=tok, chat_template=tok.chat_template)
        return model, tok, proc
