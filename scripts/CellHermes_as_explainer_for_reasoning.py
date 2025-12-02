import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from llamafactory.model import load_config, load_model, load_tokenizer
from llamafactory.hparams import get_eval_args, get_infer_args, get_train_args
from llamafactory.data import get_template_and_fix_tokenizer
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple, Union
from transformers import GenerationConfig
from llamafactory.extras.misc import get_logits_processor
from llamafactory.chat.base_engine import Response
import joblib
import json
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model_dir", required = True, help = "Path the pretrained model file")
ap.add_argument("-a","--adapter_dir", required = True, help = "Path the adapter model file")
ap.add_argument("-i","--input_text", required = True, help = "The text description of cell information and prediction results.")
ap.add_argument("-o","--output_file", required = True, help = "Path to save the output reasoning results file")
args_input = ap.parse_args()

def process_args(
    model,
    tokenizer, 
    processer,
    template, 
    generating_args, 
    messages,
    input_kwargs, 
    system=None,
):
    mm_input_dict = {"images": [], "videos": [], "imglens": [0], "vidlens": [0]}
    messages = template.mm_plugin.process_messages(
        messages, mm_input_dict["images"], mm_input_dict["videos"], processor
    )
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    system = system or generating_args["default_system"]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system)
    prompt_ids, _ = template.mm_plugin.process_token_ids(
    prompt_ids, None, mm_input_dict["images"], mm_input_dict["videos"], tokenizer, processor
    )
    prompt_length = len(prompt_ids)
    inputs = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)
    
    do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
    temperature: Optional[float] = input_kwargs.pop("temperature", None)
    top_p: Optional[float] = input_kwargs.pop("top_p", None)
    top_k: Optional[float] = input_kwargs.pop("top_k", None)
    num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
    repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
    length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
    max_length: Optional[int] = input_kwargs.pop("max_length", None)
    max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
    stop: Optional[Union[str, List[str]]] = input_kwargs.pop("stop", None)
    if stop is not None:
        logger.warning_rank0("Stop parameter is not supported by the huggingface engine yet.")
    generating_args = generating_args.copy()
    generating_args.update(
        dict(
            do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
            temperature=temperature if temperature is not None else generating_args["temperature"],
            top_p=top_p if top_p is not None else generating_args["top_p"],
            top_k=top_k if top_k is not None else generating_args["top_k"],
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else generating_args["repetition_penalty"],
            length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    
    if isinstance(num_return_sequences, int) and num_return_sequences > 1:  # do_sample needs temperature > 0
        generating_args["do_sample"] = True
        generating_args["temperature"] = generating_args["temperature"] or 1.0

    if not generating_args["temperature"]:
        generating_args["do_sample"] = False

    if not generating_args["do_sample"]:
        generating_args.pop("temperature", None)
        generating_args.pop("top_p", None)

    if max_length:
        generating_args.pop("max_new_tokens", None)
        generating_args["max_length"] = max_length

    if max_new_tokens:
        generating_args.pop("max_length", None)
        generating_args["max_new_tokens"] = max_new_tokens
    gen_kwargs = dict(
        inputs=inputs,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(**generating_args),
        logits_processor=get_logits_processor(),
    )
    mm_inputs = template.mm_plugin.get_mm_inputs(**mm_input_dict, seqlens=[prompt_ids], processor=processor)
    for key, value in mm_inputs.items():
        value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        gen_kwargs[key] = value.to(model.device)

    return gen_kwargs, prompt_length

args = {
    "model_name_or_path": f"{args_input.model_dir}",
    "adapter_name_or_path": f"{args_input.adapter_dir}",
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "do_sample": True,
    "max_new_tokens": 512, 
    "temperature": 0.95, 
    "top_p": 0.7, 
    "do_sample": True, 
    "top_k": 50
}

model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module['tokenizer']
processor = tokenizer_module['processor']
can_generate = finetuning_args.stage == "sft"
tokenizer.padding_side = "left" if can_generate else "right"
template = get_template_and_fix_tokenizer(tokenizer, data_args)
model = load_model(
    tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not can_generate)
)  # must after fixing tokenizer to resize vocab


generating_args = generating_args.to_dict()

# generate message data
messages = []
messages.append([
    {
        "role": "user",
        "content": '\n\n'+f"{args_input.input_text}"+'assistant\n\n'
    }
])

generate_outputs = []

with torch.no_grad():
    for m in tqdm(messages):
        gen_kwargs, prompt_length = process_args(model, tokenizer, processor, template, generating_args, m, input_kwargs={})
        generate_output = model.generate(**gen_kwargs)
        response = tokenizer.decode(generate_output[0][prompt_length:], skip_special_tokens=True)
        generate_outputs.append(response)

joblib.dump(generate_outputs,f"{args_input.output_file}")
