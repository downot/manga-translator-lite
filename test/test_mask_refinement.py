from manga_translator_lite.mask_refinement.text_mask_utils import _is_tiny_component_inside_line


def test_tiny_component_inside_text_line_is_kept():
    assert _is_tiny_component_inside_line(4, 0.2, 0.01)


def test_tiny_component_near_but_outside_text_line_is_rejected():
    assert not _is_tiny_component_inside_line(4, 0.0, 0.01)


def test_larger_component_does_not_use_tiny_component_rule():
    assert not _is_tiny_component_inside_line(10, 0.2, 0.01)
