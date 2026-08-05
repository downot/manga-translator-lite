import numpy as np

from manga_translator_lite.mask_refinement import _light_background_line_fill
from manga_translator_lite.utils import TextBlock


def _block_for_line(x1, y1, x2, y2):
    return TextBlock(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        texts=["..."],
        font_size=y2 - y1,
    )


def test_light_background_line_fill_covers_speech_bubble_textline():
    image = np.full((80, 80, 3), 245, dtype=np.uint8)
    block = _block_for_line(30, 20, 42, 60)

    mask = _light_background_line_fill(image, [block], light_threshold=185, padding_ratio=0.1)

    assert mask[40, 36] == 255
    assert mask.sum() > 0


def test_light_background_line_fill_skips_dark_art_regions():
    image = np.full((80, 80, 3), 95, dtype=np.uint8)
    block = _block_for_line(30, 20, 42, 60)

    mask = _light_background_line_fill(image, [block], light_threshold=185, padding_ratio=0.1)

    assert mask.sum() == 0
