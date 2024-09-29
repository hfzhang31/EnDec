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
                          LogitsProcessorList)

from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def tokens_to_ids(tokens, tokenizer):
    token_ids = []
    for token in tokens:
        id = tokenizer.encode(token)[-1]
        token_ids.append(id)

class LocalJailLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, prefix, main_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = 0
        self.prefix_tokens = prefix
        self.delta = 2.0
        self.tokenizer = tokenizer
        self.main_args = main_args
        self.prepare_prefix()
        self.prepare()
    
    def refresh(self):
        self.pos = 0
    
    def set_delta(self, delta):
        self.delta = delta

    def set_prefix(self, prefix):
        self.prefix = prefix

    def prepare_prefix(self):
        if self.prefix_tokens== None:
            self.prefix_id = []
        else:
            token_ids = []
            for token in self.prefix_tokens:
                id = self.tokenizer.encode(token)[-1]
                token_ids.append(id)
            self.prefix_id = token_ids
    
    def prepare(self):
        negative_mid_words = {'sorry':'glad', 'cannot':'can', 'illegal':'legal', 'unethical':'ethical','dangerous':'good','serious':'good'}
        self.nwid2pwid = {}
        self.nswids = []
        for k,v in negative_mid_words.items():
            nwid = self.tokenizer.encode(k)[-1]
            pwid = self.tokenizer.encode(v)[-1]
            self.nwid2pwid[nwid] = pwid
        self.ending_id = 2

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.pos < len(self.prefix_id):
            token_id = self.prefix_id[self.pos]
            scores[:,token_id] += self.delta*100
            self.pos += 1
        top_ids = torch.topk(scores,1)[1]
        # if any([x in top_ids for x in self.nswids]):
        #     scores[:,self.ending_id] += self.delta*100
        #     scores[:,self.nswids] -= self.delta*100
        # else:
        for nwid, pwid in self.nwid2pwid.items():
            if nwid in top_ids:
                scores[:, pwid] = scores[:,nwid] + self.delta*100
                scores[:, nwid] -= self.delta*100
        return scores

class LocalJail():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.processor = LocalJailLogitsProcessor(tokenizer, args.target_prefix, args)
        self.processor.set_delta(self.args.delta)
    
    def jail_processor(self):
        return self.processor
    
    def refresh(self):
        self.processor.refresh()

    def setup_prefix(self):
        ids = self.tokenizer.encode(self.args.prefix)[1:]
        self.processor.prefix_id = ids
        