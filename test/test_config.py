"""Smoke tests for the config contract — the single thing every pipeline stage
depends on. Loads config.py standalone (it has no intra-package imports) so these
run with only `pydantic` installed, no torch/cv2."""

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_config_module():
    p = REPO / "manga_translator_lite" / "config.py"
    spec = importlib.util.spec_from_file_location("mtl_config_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load_config_module()
Config = cfg.Config


def test_defaults_sane():
    c = Config()
    assert c.detector.detector.value == "default"
    assert c.ocr.ocr.value == "48px"
    assert c.translator.provider.value == "openai"
    assert c.translator.target_lang == "ENG"
    assert c.translator.concurrency == 1
    assert c.render.font_size_minimum == -1
    assert c.use_gpu is False
    assert c.detector.erase_detection_threshold == 0.0
    assert c.detector.secondary_box_fill is False


def test_load_sample_toml(tmp_path):
    sample = REPO / "config.toml.sample"
    assert sample.is_file(), "config.toml.sample is missing"
    # Config.load() dispatches on the file extension, so copy the *.sample to a real
    # .toml first — exactly what `cp config.toml.sample config.toml` does in practice.
    dest = tmp_path / "config.toml"
    dest.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    c = Config.load(str(dest))
    # The shipped sample must parse into a valid Config with sane core values.
    assert isinstance(c.translator.target_lang, str) and c.translator.target_lang
    assert c.detector.detection_size > 0
    assert c.translator.provider.value in {"openai", "gemini", "none"}
    assert c.translator.concurrency == 1


def test_font_color_helpers():
    c = Config(render={"font_color": "FFFFFF:000000"})
    assert c.render.font_color_fg == (255, 255, 255)
    assert c.render.font_color_bg == (0, 0, 0)
    assert Config().render.font_color_fg is None


def test_config_help_schema():
    # This is exactly what the `config-help` command prints.
    schema = Config.model_json_schema()
    assert "properties" in schema
    props = schema["properties"]
    for section in ("detector", "ocr", "inpainter", "translator", "render"):
        assert section in props
