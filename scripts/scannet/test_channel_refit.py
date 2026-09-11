"""Regression checks for one fixed mask fitted across all calibration scenes."""

from types import SimpleNamespace

import numpy as np
import pytest

from . import channel_refit


def test_refit_uses_all_scenes_with_equal_weight_and_nested_channels(
    tmp_path, monkeypatch
):
    # The third scene has many more pairs and opposing channel preferences.
    # Equal scene weights retain [1, 2]; either omitting it or pair weighting differs.
    losses = {"a": [4, 3, 2, -4], "b": [4, 3, 2, -4], "c": [-5, 3, 2, 4]}
    prepared = {
        name: SimpleNamespace(name=name, features=np.empty((1, 4)))
        for name in ("c", "b", "a")
    }
    seen = []

    def ablate(path, scene, channels, args, protocol):
        seen.append((scene.name, tuple(channels)))
        deleted = np.zeros((len(channels), 4))
        deleted[:, 0] = -np.asarray(losses[scene.name])[channels]
        return {
            "full": np.zeros(4),
            "deleted": deleted,
            "valid_pairs": 100 if scene.name == "c" else 1,
        }

    monkeypatch.setattr(channel_refit, "cached_ablation", ablate)
    monkeypatch.setattr(channel_refit, "scene_retrieval", lambda *args: np.zeros(4))
    channels = channel_refit.fit_fixed_mask(
        prepared,
        [3, 2],
        tmp_path,
        SimpleNamespace(),
        {"hit_distance": 0.1},
    )
    assert channels.tolist() == [1, 2]
    assert seen == [
        (name, indices)
        for indices in ((0, 1, 2, 3), (0, 1, 2))
        for name in ("a", "b", "c")
    ]


def test_refit_preserves_the_selected_pruning_schedule():
    selection = {"dimensions": [1024, 768, 512, 384, 256, 128]}
    assert channel_refit.refit_schedule(selection, 256, 1536) == [
        1024,
        768,
        512,
        384,
        256,
    ]
    with pytest.raises(ValueError, match="evaluated"):
        channel_refit.refit_schedule(selection, 200, 1536)


@pytest.mark.parametrize("schedule", [[], [2, 3], [3, 3], [4, 2], [3, 1]])
def test_refit_rejects_invalid_source_schedules(schedule):
    with pytest.raises(ValueError, match="Invalid"):
        channel_refit.refit_schedule({"dimensions": schedule}, 2, 4)
