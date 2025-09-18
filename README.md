A clean "baseline" moe transformer version to compare against Qwen3 next architecture (GDN vs No GDN essentially).


Architecture:
- Deep seek style MoE (Auxillary loss free routing: https://arxiv.org/abs/2408.15664)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Gated Attention (https://arxiv.org/abs/2505.06708)

Auxillary stuff:
- Cut cross entropy training (https://arxiv.org/abs/2411.09009)

Using uv:
```
uv sync
```

Train (trains on 5% of TinyStories-hf for 10 epochs by default)
```
uv run python main.py
```

Infer (hard coded to use checkpoint 10):
```
uv run python basic_inf.py
```


