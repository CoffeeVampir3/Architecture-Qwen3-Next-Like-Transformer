A clean "baseline" moe transformer version to compared against Qwen3 next architecture. Architcturally:

- Deep seek style MoE (Auxillary loss free routing: https://arxiv.org/abs/2408.15664)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

Using uv:
```
uv sync
```

Train (trains on wikitext-2-v1 for 10 epochs by default) 
```
uv run python main.py
```

Infer (hard coded to use checkpoint 10):
```
uv run python basic_inf.py
```



