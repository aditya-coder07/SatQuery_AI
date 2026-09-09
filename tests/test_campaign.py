"""The Phase 5 campaign harness: config validation, scheduling, and staging.

Everything here is torch-free on purpose. The failures these tests guard
against - a dependency cycle, a budget that never starts anything, a stale run
resumed twice, a corrupt shard that passes verification - all happen in path
and JSON logic, and all of them would otherwise be discovered on a cluster at
the cost of GPU hours. They belong in the no-torch CI run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from training.cluster import stage_data
from training.cluster.campaign import (
    DONE, FAILED, INTERRUPTED, PENDING, RUNNING, Campaign, RunState,
    load_campaign,
)
from training.cluster.env_probe import (
    BASE_RECIPES, GpuProfile, probe, recipe_for,
)

REPO = Path(__file__).resolve().parent.parent


def minimal_config(**overrides) -> dict:
    """A config whose scripts really exist, so validation has something to pass."""
    config = {
        "version": "test",
        "name": "test-campaign",
        "checkpoint_root": "checkpoints/test",
        "runs": [
            {"id": "a", "tool": "a_v1", "family": "encoder",
             "script": "training/track_a_full.py", "est_hours": 10},
            {"id": "b", "tool": "b_v1", "family": "change",
             "script": "training/train_change_mask.py", "depends_on": ["a"],
             "est_hours": 2},
            {"id": "c", "tool": "c_v1", "family": "grounding",
             "script": "training/train_grounding.py", "est_hours": 1},
        ],
    }
    config.update(overrides)
    return config


# --- Config validation ------------------------------------------------------


def test_real_campaign_config_is_valid(tmp_path):
    """The shipped config must load. It names ten runs and eight scripts."""
    campaign = load_campaign("configs/campaign.yaml", tmp_path)
    assert campaign.runs
    for run in campaign.runs.values():
        assert (REPO / run.script).is_file()
        # Every checkpoint directory is under v2. A run writing into a
        # published checkpoint directory would overwrite a frozen number.
        assert run.ckpt_dir.startswith("checkpoints/v2/"), run.id


def test_campaign_config_args_are_real_flags():
    """Every `--flag` in the config exists in the script it is passed to.

    A typo here costs a scheduled run that dies instantly at argparse, which
    on a shared cluster means a wasted queue slot rather than an error you see.
    """
    import re

    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        source = (REPO / run["script"]).read_text(encoding="utf-8")
        known = set(re.findall(r'add_argument\(\s*\n?\s*"(--[a-z0-9-]+)"', source))
        used = {a for a in run["args"] if a.startswith("--")}
        assert used <= known, f"{run['id']}: unknown flags {sorted(used - known)}"


def test_campaign_config_supplies_every_required_flag():
    """A run missing a `required=True` argument dies instantly at argparse.

    The regression this caught: three runs - `change_mask`, `change_caption`
    and `optsar_fusion` - were configured without `--index`, which every one
    of them requires. On a cluster that is a queue slot spent producing an
    argparse usage message. Checking that the flags used are *valid* does not
    catch it; only checking that the required ones are *present* does.
    """
    import re

    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        source = (REPO / run["script"]).read_text(encoding="utf-8")
        required = set(re.findall(
            r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*required=True', source
        ))
        used = {a for a in run["args"] if a.startswith("--")}
        assert required <= used, \
            f"{run['id']}: missing required {sorted(required - used)}"


def test_unknown_dependency_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][1]["depends_on"] = ["nonexistent"]
    with pytest.raises(ValueError, match="unknown run"):
        Campaign(config, tmp_path)


def test_dependency_cycle_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][0]["depends_on"] = ["b"]
    with pytest.raises(ValueError, match="cycle"):
        Campaign(config, tmp_path)


def test_missing_script_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][0]["script"] = "training/does_not_exist.py"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Campaign(config, tmp_path)


# --- Scheduling -------------------------------------------------------------


def test_dependencies_gate_readiness(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    assert "b" not in campaign.ready(), "b depends on a and a is not done"
    campaign.state["a"].status = DONE
    assert "b" in campaign.ready()


def test_failed_runs_are_not_picked_up_again(tmp_path):
    """A run that raised needs a human, not another GPU hour."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = FAILED
    campaign.state["c"].status = DONE
    assert campaign.next_run() not in ("a", "c")


def test_interrupted_runs_are_resumed(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = INTERRUPTED
    assert campaign.next_run() == "a"


def test_budget_prefers_a_run_that_fits(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    # a=10h, c=1h, both ready. A 90-minute session should take c.
    assert campaign.next_run(budget_minutes=90) == "c"


def test_budget_starts_a_long_run_when_nothing_fits(tmp_path):
    """The regression this exists for.

    Sessions are 4-12 hours and runs are up to 20. A budget that refused
    anything longer than the session would report "nothing ready" forever and
    train nothing at all. Every trainer checkpoints, so starting it is right.
    """
    config = minimal_config()
    for run in config["runs"]:
        run["est_hours"] = 20
    campaign = Campaign(config, tmp_path)
    assert campaign.next_run(budget_minutes=60) is not None


def test_strict_budget_refuses_instead(tmp_path):
    config = minimal_config()
    for run in config["runs"]:
        run["est_hours"] = 20
    campaign = Campaign(config, tmp_path)
    assert campaign.next_run(budget_minutes=60, strict=True) is None


def test_partly_trained_run_is_estimated_by_what_remains(tmp_path):
    """A run 90% done is a short run, not a fresh ten-hour commitment."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].seconds = 9.5 * 3600      # 30 min left of 10 h
    campaign.state["c"].status = DONE
    assert campaign.remaining_minutes("a") == pytest.approx(30, abs=1)
    assert campaign.next_run(budget_minutes=60) == "a"


# --- Crash recovery ---------------------------------------------------------


def test_stale_running_run_is_reclaimed(tmp_path):
    """The session-death case: marked running, nothing running it."""
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = "some-dead-node"
    state.owner_pid = 999999
    state.heartbeat = time.time() - 3600
    assert campaign.reclaim_stale(verbose=False) == ["a"]
    assert campaign.state["a"].status == INTERRUPTED
    assert campaign.next_run() == "a"


def test_fresh_heartbeat_is_left_alone(tmp_path):
    """A live run on another node must not be reclaimed and started twice."""
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = "another-node"
    state.owner_pid = 12345
    state.heartbeat = time.time()
    assert campaign.reclaim_stale(verbose=False) == []
    assert campaign.state["a"].status == RUNNING


def test_run_owned_by_this_live_process_is_left_alone(tmp_path):
    import os

    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None
    state.owner_pid = os.getpid()
    state.heartbeat = time.time() - 3600
    assert state.owner_alive()
    assert campaign.reclaim_stale(verbose=False) == []


def test_state_survives_a_reload(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = DONE
    campaign.state["a"].seconds = 1234.5
    campaign.save()

    reloaded = Campaign(minimal_config(), tmp_path)
    assert reloaded.state["a"].status == DONE
    assert reloaded.state["a"].seconds == 1234.5


def test_state_file_is_written_atomically(tmp_path):
    """No `.tmp` left behind, and the file parses after every save."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.save()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads(campaign.state_path.read_text(encoding="utf-8"))


def test_resume_flag_is_added_only_after_the_first_attempt(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    run = campaign.runs["a"]
    assert "--resume" not in run.command("python", resume=False)
    assert "--resume" in run.command("python", resume=True)


# --- GPU profile and batch shapes -------------------------------------------


def test_probe_never_raises_without_a_gpu():
    profile = probe(".")
    assert isinstance(profile.dtype_name, str)
    assert profile.attn_implementation in ("sdpa", "flash_attention_2")


@pytest.mark.parametrize("vram", [6.4, 16.0, 24.0, 40.0, 80.0])
@pytest.mark.parametrize("family", sorted(BASE_RECIPES))
def test_effective_batch_survives_the_card(family, vram):
    """A bigger GPU must make a run faster, not different.

    If the effective batch moved with the hardware, two runs of the same
    config on two nodes would be solving different optimisation problems and
    their metrics would not be comparable - which is exactly the thing a
    campaign spread over whatever node is free cannot afford.
    """
    profile = GpuProfile(available=True, device_count=1, name="test",
                         capability=(8, 0), vram_gb=vram, bf16=True)
    recipe = recipe_for(family, profile)
    target = BASE_RECIPES[family]["micro_batch"] * BASE_RECIPES[family]["grad_accum"]
    assert recipe["micro_batch"] >= 1
    # At or above the reference, never below it: rounding up costs a fraction
    # of a batch, rounding down silently weakens the run.
    assert target <= recipe["effective_batch"] < target * 2


def test_small_card_shrinks_the_micro_batch():
    """The bug this caught: scaling only upwards OOMs a 6 GB laptop."""
    small = GpuProfile(available=True, device_count=1, name="laptop",
                       capability=(8, 9), vram_gb=6.4, bf16=True)
    assert recipe_for("encoder", small)["micro_batch"] < \
        BASE_RECIPES["encoder"]["micro_batch"]


def test_pre_ampere_gets_fp16_and_no_flash_attention():
    turing = GpuProfile(available=True, device_count=1, name="Tesla T4",
                        capability=(7, 5), vram_gb=16.0, bf16=False)
    assert turing.dtype_name == "float16"
    assert turing.needs_grad_scaler
    assert turing.attn_implementation == "sdpa"


def test_unknown_job_family_raises():
    profile = GpuProfile()
    with pytest.raises(KeyError):
        recipe_for("not-a-family", profile)


# --- Data staging -----------------------------------------------------------


def make_dataset(root: Path, key: str = "levircd", content: bytes = b"payload") -> Path:
    subdir = root / stage_data.BY_KEY[key].subdir
    (subdir / "nested").mkdir(parents=True, exist_ok=True)
    (subdir / "a.png").write_bytes(content)
    (subdir / "nested" / "b.png").write_bytes(content + b"2")
    return subdir


def test_manifest_round_trips(tmp_path):
    make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    verdicts = stage_data.verify(tmp_path, manifest, ["levircd"])
    assert verdicts["levircd"].ok
    assert verdicts["levircd"].checked == 2


def test_verify_detects_a_missing_file(tmp_path):
    subdir = make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    (subdir / "a.png").unlink()
    verdict = stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"]
    assert not verdict.ok
    assert verdict.missing


def test_verify_detects_content_changed_at_the_same_size(tmp_path):
    """The failure size-and-mtime cannot see, and the reason digests are used.

    A zero-filled file keeps its size. `docs/model-cards.md` records twelve
    sidecars that came back from a restore as NUL bytes and a verification
    that hashed without opening them.
    """
    subdir = make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    (subdir / "a.png").write_bytes(b"\x00" * len(b"payload"))
    verdict = stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"]
    assert not verdict.ok
    assert any("digest" in c for c in verdict.corrupt)


def test_quick_manifest_is_refused_as_a_gate(tmp_path):
    make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], quick=True,
                                         progress=False)
    with pytest.raises(ValueError, match="no digests"):
        stage_data.verify(tmp_path, manifest, ["levircd"])


def test_absent_dataset_is_not_reported_ok(tmp_path):
    """An empty directory must fail, not pass vacuously."""
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    assert not stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"].ok


def test_partial_staging_reports_which_runs_are_unblocked(tmp_path):
    make_dataset(tmp_path, "levircd")
    manifest = stage_data.build_manifest(tmp_path, progress=False)
    verdicts = stage_data.verify(tmp_path, manifest)
    ready, blocked = stage_data.runs_unblocked(verdicts)
    assert "change_mask" in ready
    assert "track_a" in blocked


def test_every_campaign_run_maps_to_a_known_dataset():
    """`needs_data` must name datasets staging knows how to verify."""
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        for key in run.get("needs_data", []):
            assert key in stage_data.BY_KEY, f"{run['id']} needs unknown data '{key}'"


def test_every_required_by_names_a_real_run():
    """The other direction, which was wrong and reported runs that do not exist.

    `runs_unblocked` is read by a human deciding whether a partial transfer is
    enough to start something. Naming `stage_a2` and `caption` when neither was
    a run in the campaign made that output actively misleading.
    """
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    run_ids = {run["id"] for run in config["runs"]}
    for dataset in stage_data.DATASETS:
        for run_id in dataset.required_by:
            assert run_id in run_ids, \
                f"dataset '{dataset.key}' claims run '{run_id}', which does not exist"


def test_data_requirements_agree_in_both_directions():
    """A run that needs data X must be listed in X's `required_by`, and vice versa."""
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        for key in run.get("needs_data", []):
            assert run["id"] in stage_data.BY_KEY[key].required_by, \
                f"run '{run['id']}' needs '{key}' but is not in its required_by"
