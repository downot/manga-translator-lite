"""Input discovery tests without the extract module's ML dependencies."""

import importlib.util
import pathlib


REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_input_discovery():
    path = REPO / "manga_translator_lite" / "pipeline" / "input_discovery.py"
    spec = importlib.util.spec_from_file_location("mtl_input_discovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discovery = _load_input_discovery()


def _image(path):
    path.write_bytes(b"not decoded during discovery")
    return path


def test_single_file_is_a_single_image_task(tmp_path):
    first = _image(tmp_path / "001.png")
    _image(tmp_path / "002.png")

    tasks = discovery.discover_input_tasks(str(first))

    assert tasks == [(tmp_path.name, [str(first)])]


def test_mixed_directory_keeps_root_images_and_subtasks(tmp_path):
    root_image = _image(tmp_path / "001.png")
    chapter = tmp_path / "chapter-2"
    chapter.mkdir()
    chapter_image = _image(chapter / "002.png")

    tasks = discovery.discover_input_tasks(str(tmp_path))

    assert tasks == [
        (tmp_path.name, [str(root_image)]),
        ("chapter-2", [str(chapter_image)]),
    ]


def test_source_matches_requires_both_saved_metadata_values(tmp_path):
    image = _image(tmp_path / "001.png")
    size, mtime_ns = discovery.source_fingerprint(str(image))

    assert discovery.source_matches(str(image), size, mtime_ns)
    assert not discovery.source_matches(str(image), size, None)

    image.write_bytes(b"changed source image")
    assert not discovery.source_matches(str(image), size, mtime_ns)
