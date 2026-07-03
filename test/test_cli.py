"""Smoke tests for the core CLI command surface (extract / translate / render /
run / config-help). Loads args.py standalone — it imports only argparse/os/urllib,
so this needs no dependencies and never touches the heavy ML stack.

This covers that the commands are registered, their required args are enforced, and
options map to the expected namespace attributes that __main__._dispatch routes on.
Actually executing the pipeline needs model weights + GPU and is out of CI scope."""

import argparse
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _build_parser():
    p = REPO / "manga_translator_lite" / "args.py"
    spec = importlib.util.spec_from_file_location("mtl_args_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_parser()


def _subcommands(parser):
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return set(a.choices)
    return set()


def test_all_core_commands_registered():
    cmds = _subcommands(_build_parser())
    assert {"extract", "translate", "render", "run", "config-help"} <= cmds


def test_extract_parsing(tmp_path):
    ns = _build_parser().parse_args(
        ["extract", "-i", str(tmp_path), "-w", str(tmp_path / "work")])
    assert ns.cmd == "extract"
    assert ns.input == str(tmp_path)
    assert ns.work_dir == str(tmp_path / "work")
    assert ns.overwrite is False
    assert ns.target_lang is None and ns.verbose is False


def test_translate_parsing(tmp_path):
    ns = _build_parser().parse_args(
        ["translate", str(tmp_path), "--start-index", "5", "--overwrite", "-j", "4"])
    assert ns.cmd == "translate"
    assert ns.work_dir == str(tmp_path)
    assert ns.start_index == 5
    assert ns.overwrite is True
    assert ns.concurrency == 4
    # No reference flags → auto sentinel (None) + no_reference off.
    assert ns.reference_lang is None and ns.no_reference is False


def test_translate_reference_flags(tmp_path):
    # --reference-lang is repeatable and accumulates into a list (manual mode).
    ns = _build_parser().parse_args(
        ["translate", str(tmp_path), "--reference-lang", "CHS", "--reference-lang", "KOR"])
    assert ns.reference_lang == ["CHS", "KOR"]
    assert ns.no_reference is False
    # --no-reference is the explicit off switch.
    ns2 = _build_parser().parse_args(["translate", str(tmp_path), "--no-reference"])
    assert ns2.no_reference is True
    # run also accepts the reference flags (its translate phase uses them).
    ns3 = _build_parser().parse_args(
        ["run", "-i", str(tmp_path), "-w", str(tmp_path / "w"),
         "-o", str(tmp_path / "o"), "--reference-lang", "CHS"])
    assert ns3.reference_lang == ["CHS"]


def test_render_parsing(tmp_path):
    ns = _build_parser().parse_args(
        ["render", str(tmp_path), "-o", str(tmp_path / "out"), "--check", "-y"])
    assert ns.cmd == "render"
    assert ns.output == str(tmp_path / "out")
    assert ns.check is True and ns.yes is True and ns.no_check is False


def test_run_parsing(tmp_path):
    ns = _build_parser().parse_args(
        ["run", "-i", str(tmp_path), "-w", str(tmp_path / "w"),
         "-o", str(tmp_path / "o"), "--target-lang", "CHS", "--concurrency", "3"])
    assert ns.cmd == "run"
    assert ns.target_lang == "CHS"
    assert ns.concurrency == 3


def test_config_help_parsing():
    ns = _build_parser().parse_args(["config-help"])
    assert ns.cmd == "config-help"


def test_required_args_enforced(tmp_path):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])                          # no subcommand
    with pytest.raises(SystemExit):
        parser.parse_args(["extract"])                 # missing -i/-w
    with pytest.raises(SystemExit):
        parser.parse_args(["render", str(tmp_path)])   # missing -o
