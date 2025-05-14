import torch
from .parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
from lm_eval import evaluator
from .datautils import get_loaders
import os
from .LMClass import LMClass
from .utils import create_logger
from .categories import subcategories, categories
from tqdm import tqdm
from pprint import pprint
import torch.nn as nn
import numpy as np


@torch.no_grad()
def evaluate_ppl(model, testenc, dev):
    print("Evaluating ...")
    model.seqlen = 2048
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # logger.info(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    return ppl


@torch.no_grad()
def evaluate(lm, tasks, fewshot):
    from lm_eval.utils import make_table
    results = {}
    t_results = evaluator.simple_evaluate(
        lm,
        tasks=tasks.split(','),
        num_fewshot=fewshot,
        limit=None,
    )
    results.update(t_results)
    print(make_table(results))         
    return results


def eval_tasks(model, tokenizer, eval_ppl, model_path, tasks="", fewshot=0, batch_size="auto:4.0", seed=2):
    DEV=torch.device('cuda:0')
    if eval_ppl:
        datasets = ['wikitext2', 'c4_new'] #
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=seed, model=model_path, seqlen=2048
            )
            ppl = evaluate_ppl(model, testloader, dev=DEV)
            print(f"{dataset} Perplexity: {ppl.item():3f}")
    if tasks != "":
        from lm_eval.models.huggingface import HFLM
        model.to(DEV)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        evaluate(lm, tasks, fewshot)