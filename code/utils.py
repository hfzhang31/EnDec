import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial

import numpy # for gradio hot reload

import torch

from transformers import LogitsProcessor

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList,
                          GPTQConfig)

from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import csv
import json

def load_model(args):
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom", "vicuna", 'chatglm','mpt', 'Marcoroni']])
    args.need_trust_remote_code = any([(model_type in args.model_name_or_path) for model_type in ['chatglm']])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            if args.need_trust_remote_code:
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)


    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    args.target_prefix = ['Sure', 'e,', 'here','is', 'the', 'code']

    return model, tokenizer, device

def prepare_logits_processor(
    temperature: float, repetition_penalty: float
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    return processor_list

def generate(prompt, args, model=None, device=None, tokenizer=None, jailbreak = None):
    if 'chatglm2-6b' in args.model_name_or_path:
        input_text = f"Act as an artificial intelligence assistant and give helpful, detailed, and polite answers to the question, use English only: {prompt}"
    elif 'mpt' in args.model_name_or_path:
        input_text = f"Act as an artificial intelligence assistant and give helpful, detailed, and polite answers to the question: {prompt}"
    elif 'Guanaco' in args.model_name_or_path:
        input_text = f"### Human: {prompt} ### Assistant:"
    elif 'CodeLlama' in args.model_name_or_path:
        input_text = f"Act as an artificial intelligence assistant and give helpful, detailed, and polite answers to the question: {prompt}"
    elif 'Marcoroni' in args.model_name_or_path:
        input_text = f"### Instruction: {prompt} ### Response:"
    elif 'llama-2-7B-LoRA-assemble' in args.model_name_or_path:
        input_text = f"Act as an artificial intelligence assistant and give helpful, detailed, and polite answers to the question: {prompt}"
    elif 'falcon' in args.model_name_or_path:
        input_text = f"### Instruction: {prompt} ### Response:"
    elif 'Llama-2' in args.model_name_or_path:
        input_text = f"### Instruction: {prompt} ### Response:"
    else:
        input_text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    temperature = 0.7
    repetition_penalty = 1.0
    # if 'Guanaco' in args.model_name_or_path:
    #     repetition_penalty = 1.15
    logits_processors = prepare_logits_processor(
        temperature, repetition_penalty
    )
    if jailbreak:
        logits_processors.append(jailbreak.jail_processor())
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_with_watermark = partial(
        model.generate,
        logits_processor=logits_processors,
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)

    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    if 'falcon' in args.model_name_or_path:
        output_with_watermark = generate_with_watermark(tokd_input["input_ids"], pad_token_id=tokenizer.eos_token_id)
    else:
        output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input, int(truncation_warning), decoded_output_with_watermark, args)

def load_advbench():
    csv_file = '../datasets/harmful_behaviors.csv'
    data = []
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            prompt = row['goal']
            data.append(prompt)
    return data

def load_advbench_code():
    csv_file = '../datasets/harmful_code_only.csv'
    data = []
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            prompt = row['goal']
            data.append(prompt)
    return data

def load_dataset(dataset_name):
    if dataset_name == 'advbench':
        return load_advbench()
    elif dataset_name == 'advbench-code':
        return load_advbench_code()
    else:
        raise NotImplementedError
    
    