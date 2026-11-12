import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from transformers import T5EncoderModel, T5Config
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

def average_pool(last_hidden_states, attention_mask):
    """Mean-pool token embeddings, ignoring padding tokens."""
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def last_token_pool(last_hidden_states, attention_mask, left_padding=False):
    """Return the embedding of the last non-padding token (for decoder-style models)."""
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0)), sequence_lengths
    ]
    
def weighted_average_pool(last_hidden_states, attention_mask):
    """Position-weighted mean-pool: later tokens receive linearly higher weight."""
    seq_length = last_hidden_states.size(1)
    weights = (
        torch.arange(1, seq_length + 1, dtype=torch.float32)
        .unsqueeze(0)
        .to(last_hidden_states.device)
    )

    weighted_mask = weights * attention_mask.float()
    weighted_hidden_states = last_hidden_states * weighted_mask.unsqueeze(-1)

    sum_weighted_embeddings = torch.sum(weighted_hidden_states, dim=1)
    sum_weights = torch.sum(weighted_mask, dim=1, keepdim=True).clamp(min=1)

    weighted_average_embeddings = sum_weighted_embeddings / sum_weights

    return weighted_average_embeddings

@dataclass
class ModelConfig:
    """E.g., for T5 (encoder-decoder), only encoder is needed for hidden representations"""
    name: str
    config_class: Optional[any] = None
    model_class: Optional[any] = None
    dropout_field: str = "dropout_rate"

MODEL_CONFIGS = {
    "t5-base": ModelConfig(
        "t5-base", 
        T5Config, 
        T5EncoderModel
    ),
    
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    )
}

from utils.context_manager import CACHE_DIR, CACHE_ONLY

def get_model_and_tokenizer(model_name: str, device: str='cpu', local_files_only=CACHE_ONLY):
    """Load a tokenizer and model from the local ./from_pretrained cache (no network access).

    T5-family models are loaded via T5EncoderModel (no decoder) with dropout zeroed for deterministic inference.
    """
    try: # NOTE, use_fast=True causes "process forking" warning for ChemBERTa
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=local_files_only, use_fast=False
        )
    except: # NOTE For NaCh0, use_fast=True is necessary
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=local_files_only, use_fast=True
        )
    
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=local_files_only)
        setattr(config, model_config.dropout_field, 0)
        model = model_config.model_class.from_pretrained(
            model_name, config=config, cache_dir=CACHE_DIR, local_files_only=local_files_only
        ).to(device)
    else:
        # NOTE To mute warning from loading ChemBERTa
        if 'chemberta' in model_name.lower():
            model = AutoModelForMaskedLM.from_pretrained(model_name, device_map=device, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=local_files_only)
        # NOTE For NaCh0, get full model with decoder, to work with prompt
        elif 'nach0' in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=local_files_only)
        else:
            model = AutoModel.from_pretrained(model_name, device_map=device, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=local_files_only)

    return model, tokenizer