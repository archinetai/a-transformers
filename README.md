
# A-Transformers

A collection of transformer models, in PyTorch.

```bash
pip install a-transformers
```
[![PyPI - Python Version](https://img.shields.io/pypi/v/a-transformers?style=flat&colorA=black&colorB=black)](https://pypi.org/project/a-transformers/)


## Usage

### Transformer
```python
from a_transformers.transformers import Transformer

transformer = Transformer(
    features=768,
    max_length=256,
    num_layers=12,
    head_features=64,
    num_heads=12,
    multiplier=4
)

x = torch.randn(2, 12, 768)
y = transformer(x) # [2, 12, 768]
```

### Resampler
```python
from a_transformers.transformers import Resampler

resampler = Resampler(
    features=768,
    in_tokens=12,
    out_tokens=4,
    num_layers=12,
    head_features=64,
    num_heads=12,
    multiplier=4
)

x = torch.randn(2, 12, 768)
y = resampler(x) # [2, 4, 768]
```

### RQ-Transformer
```python
from a_transformers.rq_transformer import RQTransformer

num_residuals = 4
codebook_size = 2048

rqtransformer = RQTransformer(
    features=768,
    max_length=64,
    max_residuals=num_residuals,
    num_tokens=codebook_size,
    num_layers=8,
    head_features=64,
    num_heads=8,
    multiplier=4,
    shared_codebook=False
)

# Training
x = torch.randint(0, 2048, (1, 64, num_residuals)) # [b, t, r]
loss = rqtransformer(x) # tensor(9.399146, grad_fn=<NllLoss2DBackward0>)

# Genration
sequence = rqtransformer.generate(x, sequence_length=64) # [1, 64, 4]
```
