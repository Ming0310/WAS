## Weight-Aware Activation Sparsity with Constrained Bayesian Optimization Scheduling for Large Language Models

### Abstract

Activation sparsity provides a dynamic, input-dependent alternative to weight pruning for accelerating inference in large language models (LLMs), effectively reducing unnecessary computations and memory accesses during the forward pass. Despite its promise, existing activation sparsification methods suffer from two major limitations: (1) solely relying on activation magnitude for sparsification, ignoring the coupling influence with the corresponding weights, (2) applying uniform sparsity rates across all blocks without considering block-wise sparsity sensitivity. To address these issues, this paper proposes a novel training-free weight-aware activation sparsity framework, called WAS. Firstly, with analyzing the coupling relationshape between weight and activation, we introduce a weight-aware scoring method to measure the activation importance in sparsification. Then, a novel constrained Bayesian optimization algorithm is further devised to set a suitable sparsity ratio for all blocks based on the sparsity sensitivity. Finally, we implement a custom GPU sparsity kernel to support the resulting sparsity patterns for wall-clock decoding speed-ups. Our WAS achieves competitive performance at 60\% model-level sparsity and significantly outperforms prior methods at higher sparsity levels, achieving up to 1.68× inference speed-up—at no retraining or weight update.

### Reproduce results

Download huggingface models and run [reproduce.ipynb](reproduce.ipynb)
