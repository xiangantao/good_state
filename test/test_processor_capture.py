"""Integration tests for capture points in model-specific processors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
from torch import Tensor, nn
from torch.nn import functional as F

from heft.attn_hook import (
    AttentionFeatureCapture,
    AttentionProcessorKind,
    CaptureContext,
    CapturedFeature,
    CaptureSpec,
    FeatureKind,
)


class CollectingSink:
    def __init__(self) -> None:
        self.records: list[CapturedFeature] = []

    def __call__(self, feature: CapturedFeature) -> None:
        self.records.append(feature)


class Clone(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.clone()


def _reshape_heads(tensor: Tensor, heads: int) -> Tensor:
    return tensor.unflatten(2, (heads, -1)).transpose(1, 2)


def _rotate_pairs(tensor: Tensor) -> Tensor:
    real, imaginary = tensor.reshape(*tensor.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-imaginary, real), dim=-1).flatten(3)


def _fake_attention(*, heads: int = 2) -> Any:
    return SimpleNamespace(
        add_k_proj=None,
        heads=heads,
        is_cross_attention=False,
        norm_k=None,
        norm_q=None,
        prepare_attention_mask=lambda mask, *_: mask,
        to_k=Clone(),
        to_out=(nn.Identity(), nn.Identity()),
        to_q=Clone(),
        to_v=Clone(),
    )


def _install_capture(
    attention: Any, processor: AttentionProcessorKind
) -> CollectingSink:
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(
            step=0,
            layers=[0],
            features=[FeatureKind.QUERY, FeatureKind.KEY, FeatureKind.HIDDEN_STATES],
        ),
        context=CaptureContext(),
        sink=sink,
    )
    session = capture.attach([SimpleNamespace(attn1=attention)], processor=processor)
    session.begin_chunk(chunk=0)
    return sink


def _captured_features(sink: CollectingSink) -> dict[FeatureKind, Tensor]:
    features: dict[FeatureKind, Tensor] = {}
    for kind in FeatureKind:
        records = sorted(
            (record for record in sink.records if record.kind is kind),
            key=lambda record: record.head,
        )
        if records:
            features[kind] = torch.stack([record.tensor for record in records], dim=1)
    return features


def test_wan_processor_exposes_rope_query_key_and_per_head_output() -> None:
    attention = _fake_attention()
    sink = _install_capture(attention, AttentionProcessorKind.WAN)
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    rotary_emb = torch.full((1, 1, 2, 1), 1j, dtype=torch.complex128)

    output = WanAttnProcessor2_0()(attention, hidden_states, rotary_emb=rotary_emb)
    features = _captured_features(sink)

    unrotated = _reshape_heads(hidden_states, attention.heads)
    expected_query = _rotate_pairs(unrotated)
    expected_hidden_states = F.scaled_dot_product_attention(
        expected_query,
        expected_query,
        unrotated,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(features[FeatureKind.QUERY], expected_query)
    torch.testing.assert_close(features[FeatureKind.KEY], expected_query)
    torch.testing.assert_close(
        features[FeatureKind.HIDDEN_STATES], expected_hidden_states
    )
    torch.testing.assert_close(
        output, expected_hidden_states.transpose(1, 2).flatten(2, 3)
    )


def test_cosmos_processor_exposes_rope_query_key_and_per_head_output() -> None:
    attention = _fake_attention()
    sink = _install_capture(attention, AttentionProcessorKind.COSMOS)
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    rotary_emb: Any = (torch.zeros(2, 2), torch.ones(2, 2))

    output = CosmosAttnProcessor2_0()(
        attention, hidden_states, image_rotary_emb=rotary_emb
    )
    features = _captured_features(sink)

    unrotated = _reshape_heads(hidden_states, attention.heads)
    expected_query = _rotate_pairs(unrotated)
    expected_hidden_states = F.scaled_dot_product_attention(
        expected_query,
        expected_query,
        unrotated,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(features[FeatureKind.QUERY], expected_query)
    torch.testing.assert_close(features[FeatureKind.KEY], expected_query)
    torch.testing.assert_close(
        features[FeatureKind.HIDDEN_STATES], expected_hidden_states
    )
    torch.testing.assert_close(
        output, expected_hidden_states.transpose(1, 2).flatten(2, 3)
    )


def test_cogvideox_processor_exposes_only_rope_video_tokens() -> None:
    attention = _fake_attention()
    sink = _install_capture(attention, AttentionProcessorKind.COGVIDEOX)
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()
    text_states = torch.tensor(
        [
            [[-9.0, -10.0, -11.0, -12.0]],
            [[9.0, 10.0, 11.0, 12.0]],
        ]
    )
    video_states = torch.tensor(
        [
            [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        ]
    )
    rotary_emb: Any = (torch.zeros(2, 2), torch.ones(2, 2))

    output, _ = CogVideoXAttnProcessor2_0()(
        attention,
        video_states,
        text_states,
        image_rotary_emb=rotary_emb,
    )
    features = _captured_features(sink)

    combined = torch.cat((text_states, video_states), dim=1)
    unrotated = _reshape_heads(combined, attention.heads)
    rotated_video = _rotate_pairs(unrotated[:, :, 1:])
    rotated_combined = torch.cat((unrotated[:, :, :1], rotated_video), dim=2)
    all_hidden_states = F.scaled_dot_product_attention(
        rotated_combined,
        rotated_combined,
        unrotated,
        dropout_p=0.0,
        is_causal=False,
    )[:, :, 1:]
    expected_query = rotated_video[1:]
    expected_hidden_states = all_hidden_states[1:]
    torch.testing.assert_close(features[FeatureKind.QUERY], expected_query)
    torch.testing.assert_close(features[FeatureKind.KEY], expected_query)
    torch.testing.assert_close(
        features[FeatureKind.HIDDEN_STATES], expected_hidden_states
    )
    torch.testing.assert_close(output, all_hidden_states.transpose(1, 2).flatten(2, 3))
