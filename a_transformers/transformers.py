from inspect import isfunction
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import Tensor, einsum, nn
from typing_extensions import TypeGuard

T = TypeVar("T")

"""
Utils
"""


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log(val: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(val.clamp(min=eps))


def gumbel_sample(val: Tensor, temperature: float, dim: int = -1) -> Tensor:
    noise = torch.zeros_like(val).uniform_(0, 1)
    gumbel_noise = -log(-log(noise))
    return ((val / temperature) + gumbel_noise).argmax(dim=dim)


def top_k(logits: Tensor, threshold: float) -> Tensor:
    num_logits = logits.shape[-1]
    k = max(int((1 - threshold) * num_logits), 1)
    values, indices = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, indices, values)
    return probs


"""
Attention Components
"""


def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = default(out_features, features)

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features, bias=False
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        # Compute similarity matrix and add eventual mask
        sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
        sim = add_mask(sim, mask) if exists(mask) else sim

        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def sequential_mask(mask: Tensor) -> Tensor:
    return rearrange(mask, "b j -> b 1 j")


def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask


def cross_mask(q_mask: Tensor, k_mask: Tensor):
    q_mask = rearrange(q_mask, "b i -> b i 1")
    k_mask = rearrange(k_mask, "b j -> b 1 j")
    mask = q_mask * k_mask
    return mask


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.context_features = context_features
        self.causal = causal
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,  # [b, n], false is masked
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked
        attention_mask: Optional[Tensor] = None,  # [b, n, m], false is masked
    ) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message

        # Use context if provided
        context = default(context, x)

        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        # Compute attention mask (only one is applied, use attention_mask for custom)
        if exists(mask) and not exists(context_mask):
            attention_mask = sequential_mask(mask)
        if exists(mask) and exists(context_mask):
            attention_mask = cross_mask(mask, context_mask)
        if self.causal:
            attention_mask = causal_mask(q, k)

        # Compute and return attention
        return self.attention(q, k, v, mask=attention_mask)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, features: int, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        length, device = x.shape[1], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        return self.embedding(position)


"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        multiplier: int,
        causal: bool = False,
        max_length: Optional[int] = None,
        use_positional_embedding: bool = False,
        use_attention: bool = True,
        use_cross_attention: bool = False,
        context_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_positional_embedding = use_positional_embedding
        self.use_cross_attention = use_cross_attention
        self.use_attention = use_attention

        if use_positional_embedding:
            assert_message = "max_length required if use_positional_embedding=True"
            assert exists(max_length), assert_message

            self.positional_embedding = AbsolutePositionalEmbedding(
                max_length=max_length,
                features=features,
            )

        if use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                head_features=head_features,
                num_heads=num_heads,
                context_features=context_features,
            )

        if use_attention:
            self.attention = Attention(
                features=features,
                head_features=head_features,
                num_heads=num_heads,
                causal=causal,
                out_features=out_features,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_positional_embedding:
            x = self.positional_embedding(x) + x

        if self.use_attention:
            x = self.attention(x, mask=mask, attention_mask=attention_mask) + x

        if self.use_cross_attention:
            x = (
                self.cross_attention(
                    x,
                    mask=mask,
                    context=context,
                    context_mask=context_mask,
                    attention_mask=cross_attention_mask,
                )
                + x
            )

        return self.feed_forward(x) + x


"""
Transformers
"""


class Transformer(nn.Module):
    def __init__(
        self,
        features: int,
        max_length: int,
        num_layers: int,
        head_features: int,
        num_heads: int,
        multiplier: int,
        causal: bool = False,
        use_positional_embedding: bool = True,
        use_attention: bool = True,
        use_cross_attention: bool = False,
        context_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.features = features
        self.max_length = max_length

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    causal=causal,
                    max_length=max_length,
                    use_positional_embedding=use_positional_embedding,
                    use_attention=use_attention,
                    use_cross_attention=use_cross_attention,
                    context_features=context_features,
                    out_features=out_features,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, embedding: Tensor, **kwargs) -> Tensor:
        for block in self.blocks:
            embedding = block(embedding, **kwargs)
        return embedding


class TransformerShifter(nn.Module):
    def __init__(self, transformer: Transformer, *, num_shift: int = 1):
        super().__init__()
        self.features = transformer.features
        self.max_length = transformer.max_length
        self.transformer = transformer
        self.num_shift = num_shift
        self.tokens = nn.Parameter(torch.randn(num_shift, self.features))

    def forward(self, embedding: Tensor, **kwargs) -> Tensor:
        b = embedding.shape[0]
        embedding_head = embedding[:, self.num_shift :]
        embedding_tail = repeat(self.tokens, "n d -> b n d", b=b)
        embedding = torch.cat([embedding_head, embedding_tail], dim=-2)
        embedding = self.transformer(embedding, **kwargs)
        return embedding


class Resampler(Transformer):
    def __init__(
        self,
        features: int,
        in_tokens: int,
        out_tokens: int,
        use_random: bool = False,
        **kwargs,
    ):
        super().__init__(
            features=features,
            max_length=out_tokens,
            causal=False,
            use_cross_attention=True,
            use_positional_embedding=True,
            **kwargs,
        )
        self.embedding = nn.Parameter(torch.randn(out_tokens, features))
        self.context_positional_embedding = AbsolutePositionalEmbedding(
            max_length=in_tokens,
            features=features,
        )
        self.use_random = use_random

    def forward(  # type: ignore
        self, context: Tensor, use_random: Optional[bool] = None, **kwargs
    ) -> Tensor:
        b, n, device = context.shape[0], self.embedding.shape[0], context.device
        use_random = default(use_random, self.use_random)
        context = context + self.context_positional_embedding(context)
        if use_random:
            assert self.use_random, "use_random requires use_random=True at init"
            indices = torch.randint(0, n, size=(b * n,), device=device)
            embedding = rearrange(self.embedding[indices], "(b n) d -> b n d", b=b)
        else:
            embedding = repeat(self.embedding, "n d -> b n d", b=b)
        return super().forward(embedding, context=context, **kwargs)


class Autoregressive(nn.Module):
    def __init__(self, transformer: Transformer, num_tokens: int, **kwargs):
        super().__init__()
        self.features = transformer.features
        self.max_length = transformer.max_length
        self.transformer = transformer

        self.to_embedding = nn.Embedding(num_tokens, self.features)
        self.to_logits = nn.Linear(in_features=self.features, out_features=num_tokens)

    def compute_logits(self, tokens: Tensor, **kwargs) -> Tensor:
        input_embedding = self.to_embedding(tokens)
        # Compute output embedding and logits
        output_embedding = self.transformer(input_embedding, **kwargs)
        output_logits = self.to_logits(output_embedding)
        return output_logits

    def forward(self, tokens: Tensor, **kwargs) -> Tensor:
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        logits = self.compute_logits(input_tokens)
        logits = rearrange(logits, "b n t -> b t n")
        loss = F.cross_entropy(logits, target_tokens)
        return loss

    def generate(
        self,
        start_tokens: Tensor,
        sequence_length: int,
        top_k_threshold: float = 0.9,
        temperature: float = 1.0,
        keep_start: bool = False,
        **kwargs,
    ) -> Tensor:
        t, s = start_tokens.shape[1], self.max_length
        tokens = start_tokens

        for _ in range(sequence_length):
            # Compute last token logits
            logits = self.compute_logits(tokens=tokens[:, -s:], **kwargs)
            logits = logits[:, -1]
            # Gumbel sample from top-k logits
            logits = top_k(logits, threshold=top_k_threshold)
            sample = gumbel_sample(logits, dim=-1, temperature=temperature)
            # Append sampled token
            tokens = torch.cat([tokens, rearrange(sample, "b -> b 1")], dim=-1)

        return tokens if keep_start else tokens[:, t:]


class ContinuousAutoregressive(nn.Module):
    def __init__(self, transformer: Transformer, **kwargs):
        super().__init__()
        self.transformer = transformer
        self.max_length = transformer.max_length

    def forward(self, embedding: Tensor, **kwargs) -> Tensor:
        input_embedding = embedding[:, :-1, :]
        target_embedding = embedding[:, 1:, :]
        output_embedding = self.transformer(input_embedding, **kwargs)
        return F.mse_loss(target_embedding, output_embedding)

    def generate(
        self,
        start_embedding: Tensor,
        sequence_length: int,
        keep_start: bool = False,
        **kwargs,
    ) -> Tensor:
        t, s = start_embedding.shape[1], self.max_length
        embedding = start_embedding

        for _ in range(sequence_length):
            output_embedding = self.transformer(embedding[:, -s:, :], **kwargs)
            output_embedding_last = rearrange(
                output_embedding[:, -1, :], "b d -> b 1 d"
            )
            embedding = torch.cat([embedding, output_embedding_last], dim=-2)

        return embedding if keep_start else embedding[:, t:, :]
