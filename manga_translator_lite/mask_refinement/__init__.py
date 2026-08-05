from typing import List
import cv2
import numpy as np

from .text_mask_utils import complete_mask_fill, complete_mask
from ..utils import TextBlock, Quadrilateral
from ..utils.bubble import is_ignore


def _light_background_line_fill(
    raw_image: np.ndarray,
    text_regions: List[TextBlock],
    light_threshold: int = 185,
    padding_ratio: float = 0.18,
) -> np.ndarray:
    """Fill detected text-line boxes only when their local background is bubble-like.

    Stroke masks can miss tiny vertical punctuation or lightly printed kana, leaving
    small stains after inpainting. Speech bubbles are usually locally bright and
    low-detail, so a filled line polygon is a safe fallback there; on darker art or
    screentones we skip it to avoid rectangular over-erasure.
    """
    h, w = raw_image.shape[:2]
    fallback = np.zeros((h, w), dtype=np.uint8)
    if not text_regions:
        return fallback

    gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY) if raw_image.ndim == 3 else raw_image
    light_threshold = int(np.clip(light_threshold, 0, 255))
    padding_ratio = max(float(padding_ratio), 0.0)

    for region in text_regions:
        for line in region.lines:
            pts = np.asarray(line, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            x1 = max(int(np.floor(np.min(pts[:, 0]))), 0)
            y1 = max(int(np.floor(np.min(pts[:, 1]))), 0)
            x2 = min(int(np.ceil(np.max(pts[:, 0]))), w - 1)
            y2 = min(int(np.ceil(np.max(pts[:, 1]))), h - 1)
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            if bw <= 1 or bh <= 1:
                continue

            pad = max(2, int(round(min(bw, bh) * 0.35)))
            cx1 = max(x1 - pad, 0)
            cy1 = max(y1 - pad, 0)
            cx2 = min(x2 + pad, w - 1)
            cy2 = min(y2 + pad, h - 1)
            patch = gray[cy1:cy2 + 1, cx1:cx2 + 1]
            if patch.size == 0:
                continue

            median = float(np.median(patch))
            light_ratio = float(np.mean(patch >= light_threshold))
            if median < light_threshold and light_ratio < 0.55:
                continue

            line_mask = np.zeros_like(fallback)
            cv2.fillConvexPoly(line_mask, np.round(pts).astype(np.int32), 255)
            dilate = max(1, int(round(min(bw, bh) * padding_ratio)))
            if dilate > 0:
                k = dilate * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                line_mask = cv2.dilate(line_mask, kernel, iterations=1)
            fallback = cv2.bitwise_or(fallback, line_mask)

    return fallback


async def dispatch(
    text_regions: List[TextBlock],
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    method: str = 'fit_text',
    dilation_offset: int = 0,
    ignore_bubble: int = 0,
    verbose: bool = False,
    kernel_size: int = 3,
    bubble_text_fill: bool = True,
    bubble_text_fill_threshold: int = 185,
    bubble_text_fill_padding: float = 0.18,
) -> np.ndarray:
    # Larger sized mask images will probably have crisper and thinner mask segments due to being able to fit the text pixels better
    # so we dont want to size them down as much to not lose information
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)

    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)

    mask_resized[mask_resized > 0] = 255
    textlines = []
    for region in text_regions:
        for l in region.lines:
            q = Quadrilateral(l * scale_factor, '', 0)
            textlines.append(q)

    final_mask = (
        complete_mask(
            img_resized,
            mask_resized,
            textlines,
            dilation_offset=dilation_offset,
            kernel_size=kernel_size,
        )
        if method == 'fit_text'
        else complete_mask_fill(mask_resized.shape, [txtln.aabb.xywh for txtln in textlines])
    )
    if final_mask is None:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    else:
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255

    if bubble_text_fill:
        fallback_mask = _light_background_line_fill(
            raw_image,
            text_regions,
            bubble_text_fill_threshold,
            bubble_text_fill_padding,
        )
        final_mask = cv2.bitwise_or(final_mask, fallback_mask)

    if ignore_bubble < 1 or ignore_bubble > 50:
        return final_mask

    # bubble
    kernel_size = int(max(final_mask.shape) * 0.025)  # 选择一个合适的核大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)  # 根据需要调整迭代次数
    # border
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        temp_mask = np.zeros_like(final_mask)
        # rect min
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
        # get textblock
        textblock=cv2.bitwise_and(raw_image, raw_image, mask=temp_mask)
        if is_ignore(textblock, ignore_bubble):
            cv2.drawContours(final_mask, [cnt], -1, 0, -1)

    return final_mask
