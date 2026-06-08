import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *


def load_gpt_model_and_tokenizer(model_name:str, device='cuda', revision=None):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'
    
    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names
    
    """
    assert model_name is not None

    print("Loading: ", model_name)

    if model_name == 'gpt2-xl':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "head_dim":model.config.n_embd // model.config.n_head,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "head_dim":model.config.n_embd // model.config.n_head,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)],
                      "prepend_bos":False}

    elif 'gpt-neox' in model_name.lower() or 'pythia' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        if revision is not None and 'pythia' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, revision=revision).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "head_dim":model.config.hidden_size // model.config.num_attention_heads,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                      "prepend_bos":False}
        
    elif 'gemma' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)

        # Gemma-3 multimodal (4B+) loads as Gemma3ForConditionalGeneration: text layers at model.language_model.layers
        # Gemma-3 1B text-only loads as Gemma3ForCausalLM: text layers at model.layers
        # Gemma-1/2 load as GemmaForCausalLM: text layers at model.layers
        if hasattr(model, 'language_model'):
            _layer_prefix = 'language_model.model'
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            _layer_prefix = 'model.language_model'
        else:
            _layer_prefix = 'model'
        _cfg = getattr(model.config, 'text_config', model.config)
        # Gemma-3 has an explicit head_dim that differs from hidden_size / num_attention_heads
        _head_dim = getattr(_cfg, 'head_dim', _cfg.hidden_size // _cfg.num_attention_heads)
        MODEL_CONFIG={"n_heads":_cfg.num_attention_heads,
                      "n_layers":_cfg.num_hidden_layers,
                      "resid_dim":_cfg.hidden_size,
                      "head_dim":_head_dim,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'{_layer_prefix}.layers.{layer}.self_attn.o_proj' for layer in range(_cfg.num_hidden_layers)],
                      "layer_hook_names":[f'{_layer_prefix}.layers.{layer}' for layer in range(_cfg.num_hidden_layers)],
                      "prepend_bos":True}
        
    elif 'llama' in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config
            )
        else:
            if '7b' in model_name.lower() or '8b' in model_name.lower():
                model_dtype = torch.float32
            else: #half precision for bigger llama models
                model_dtype = torch.float16
            
            # If transformers version is < 4.31 use LlamaLoaders
            # tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

            # If transformers version is >= 4.31, use AutoLoaders
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "head_dim":model.config.hidden_size // model.config.num_attention_heads,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                      "prepend_bos":True}
    elif "olmo" in model_name.lower():
        
        model_dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if revision is not None:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, revision=revision).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "head_dim":model.config.hidden_size // model.config.num_attention_heads,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                      "prepend_bos":False}
    elif 'qwen' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

        MODEL_CONFIG = {"n_heads":         model.config.num_attention_heads,
                        "n_layers":        model.config.num_hidden_layers,
                        "resid_dim":       model.config.hidden_size,
                        "head_dim":        model.config.hidden_size // model.config.num_attention_heads,
                        "name_or_path":    model.config._name_or_path,
                        "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                        "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                        "prepend_bos":     False}

    else:
        raise NotImplementedError("Still working to get this model available!")

    
    return model, tokenizer, MODEL_CONFIG

def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)