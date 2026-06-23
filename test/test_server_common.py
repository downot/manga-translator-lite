"""Security-contract tests for server_common — the path-safety + pipeline-dispatch
base shared by server.py (OSS) and downstream editor servers.

These functions are the only thing standing between a client-supplied path/lang and
the filesystem, so they get explicit traversal/escape coverage. server_common is a
plain top-level module with stdlib-only imports, so this runs without torch/aiohttp."""

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load():
    p = REPO / "server_common.py"
    spec = importlib.util.spec_from_file_location("server_common_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load()


# ---- confine: stays inside the task dir -------------------------------------
def test_confine_allows_inside(tmp_path):
    (tmp_path / "in").mkdir()
    got = sc.confine("in/page.png", tmp_path)
    assert got is not None and got == (tmp_path / "in" / "page.png")


def test_confine_rejects_parent_traversal(tmp_path):
    assert sc.confine("../../etc/passwd", tmp_path) is None


def test_confine_rejects_absolute_outside(tmp_path):
    assert sc.confine("/etc/passwd", tmp_path) is None


def test_confine_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_secret"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside)
    # resolve() follows the symlink before the containment check, so this escapes.
    assert sc.confine("link/secret.txt", tmp_path) is None


# ---- resolve_within: allowed bases + protected names ------------------------
def test_resolve_within_allows_base(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("x = 1")
    assert sc.resolve_within(str(f), [tmp_path]) == f


def test_resolve_within_blocks_protected_names(tmp_path):
    for name in ("sessions.json", "access.json", ".env", ".secret"):
        p = tmp_path / name
        p.write_text("secret")
        assert sc.resolve_within(str(p), [tmp_path]) is None


def test_resolve_within_rejects_outside_all_bases(tmp_path):
    other = tmp_path.parent / "elsewhere.toml"
    other.write_text("x = 1")
    assert sc.resolve_within(str(other), [tmp_path]) is None


# ---- is_safe_lang -----------------------------------------------------------
def test_is_safe_lang_accepts_codes():
    for ok in ("en", "zh-Hans", "pt_BR", "CHS"):
        assert sc.is_safe_lang(ok)


def test_is_safe_lang_rejects_traversal_and_seps():
    for bad in ("../x", "a/b", "a\\b", "", "a.json", "a b"):
        assert not sc.is_safe_lang(bad)


# ---- parse_reference_langs --------------------------------------------------
def test_parse_reference_langs():
    assert sc.parse_reference_langs("auto") is None
    assert sc.parse_reference_langs("off") == []
    assert sc.parse_reference_langs(["en", "zh"]) == ["en", "zh"]


# ---- resolve_pipeline_paths -------------------------------------------------
def test_resolve_pipeline_paths_rejects_bad_input(tmp_path):
    try:
        sc.resolve_pipeline_paths("../escape", None, None, tmp_path, tmp_path)
        assert False, "expected PipelinePathError"
    except sc.PipelinePathError as e:
        assert "input" in str(e).lower()


def test_resolve_pipeline_paths_rejects_missing_config(tmp_path):
    try:
        sc.resolve_pipeline_paths(None, None, "config.toml", tmp_path, tmp_path)
        assert False, "expected PipelinePathError"
    except sc.PipelinePathError as e:
        assert "config" in str(e).lower()


def test_resolve_pipeline_paths_normalizes_good_paths(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "config.toml").write_text("x = 1")
    inp, out, cfg = sc.resolve_pipeline_paths("in", "out", "config.toml", tmp_path, tmp_path)
    assert inp.endswith("/in") and cfg.endswith("/config.toml")
