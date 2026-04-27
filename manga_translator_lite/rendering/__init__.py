import os
import cv2
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from . import text_render
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

def _text_fits_region(font_size: int, text: str, max_w: float, max_h: float,
                      is_horizontal: bool, lang: str = 'en_US',
                      line_spacing_ratio: float = 0.01) -> bool:
    """Check whether text at a given font_size fits inside max_w × max_h."""
    if font_size < 1:
        return True
    spacing = int(font_size * line_spacing_ratio)

    if is_horizontal:
        line_text_list, line_width_list = text_render.calc_horizontal(
            font_size, text, max_width=int(max_w), max_height=int(max_h), language=lang
        )
        total_h = font_size * len(line_text_list) + spacing * max(len(line_text_list) - 1, 0)
        max_line_w = max(line_width_list) if line_width_list else 0
        return total_h <= max_h and max_line_w <= max_w * 1.05  # 5% tolerance on width
    else:
        line_text_list, line_height_list = text_render.calc_vertical(
            font_size, text, max_height=int(max_h)
        )
        total_w = font_size * len(line_text_list) + spacing * max(len(line_text_list) - 1, 0)
        max_line_h = max(line_height_list) if line_height_list else 0
        return total_w <= max_w and max_line_h <= max_h * 1.05


def _find_optimal_font_size(text: str, max_w: float, max_h: float,
                            is_horizontal: bool, font_size_min: int,
                            font_size_max: int, lang: str = 'en_US',
                            line_spacing_ratio: float = 0.01) -> int:
    """Binary-search for the largest font size that still fits inside the region."""
    lo, hi = font_size_min, font_size_max
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _text_fits_region(mid, text, max_w, max_h, is_horizontal, lang, line_spacing_ratio):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'],
                                font_size_fixed: int, font_size_offset: int,
                                font_size_minimum: int):
    """
    For each text region, find the optimal font size that fills the original
    detected area as completely as possible, then return the (possibly expanded)
    destination polygon for rendering.

    Strategy (in priority order):
    1. **Fit-to-region** – binary-search for the largest font size ≤ the OCR
       font size (+ offset) that makes the translated text fit inside the
       original bounding box.  This keeps the output clean by respecting the
       inpainted area.
    2. **Expand box** – if the text cannot fit even at ``font_size_minimum``,
       the bounding box is expanded just enough to contain the text.

    Returns:
        List of destination point arrays (one per region).
    """
    # Auto-calculate minimum font size from image dimensions
    if font_size_minimum == -1:
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    font_size_minimum = max(1, font_size_minimum)

    dst_points_list = []

    for region in text_regions:
        # --- 1. Determine the font-size search range -----------------------
        original_fs = region.font_size if region.font_size > 0 else font_size_minimum

        if font_size_fixed is not None:
            fs_upper = font_size_fixed
        else:
            fs_upper = original_fs + font_size_offset
        fs_upper = max(fs_upper, font_size_minimum, 1)

        # Region dimensions (unrotated)
        max_w, max_h = region.unrotated_size
        is_horiz = region.horizontal
        lang = getattr(region, "target_lang", "en_US")
        line_sp = region.line_spacing if hasattr(region, 'line_spacing') else 0.01

        # --- 2. Binary-search for optimal font size -----------------------
        if max_w > 0 and max_h > 0:
            # Allow searching up to 120 % of original font size so that short
            # translations can use a slightly bigger font and fill the box.
            search_upper = int(fs_upper * 1.2)
            target_font_size = _find_optimal_font_size(
                region.translation, max_w, max_h,
                is_horiz, font_size_minimum, search_upper, lang, line_sp or 0.01
            )
        else:
            target_font_size = fs_upper

        # --- 3. Decide whether to keep or expand the box ------------------
        fits = _text_fits_region(
            target_font_size, region.translation, max_w, max_h,
            is_horiz, lang, line_sp or 0.01
        ) if max_w > 0 and max_h > 0 else True

        dst_points = None

        if fits:
            # Text fits → use the original bounding box (no expansion).
            dst_points = region.min_rect
        else:
            # Text does NOT fit even at font_size_minimum.
            # Expand the box just enough.
            if is_horiz:
                line_text_list, line_width_list = text_render.calc_horizontal(
                    target_font_size, region.translation,
                    max_width=int(max_w), max_height=int(max_h), language=lang
                )
                needed_h = (target_font_size * len(line_text_list)
                            + int(target_font_size * (line_sp or 0.01))
                            * max(len(line_text_list) - 1, 0))
                needed_w = max(line_width_list) if line_width_list else max_w
                scale_y = max(needed_h / max_h, 1.0) if max_h > 0 else 1.0
                scale_x = max(needed_w / max_w, 1.0) if max_w > 0 else 1.0
            else:
                line_text_list, line_height_list = text_render.calc_vertical(
                    target_font_size, region.translation, max_height=int(max_h)
                )
                needed_w = (target_font_size * len(line_text_list)
                            + int(target_font_size * (line_sp or 0.01))
                            * max(len(line_text_list) - 1, 0))
                needed_h = max(line_height_list) if line_height_list else max_h
                scale_x = max(needed_w / max_w, 1.0) if max_w > 0 else 1.0
                scale_y = max(needed_h / max_h, 1.0) if max_h > 0 else 1.0

            # Cap expansion at 1.5× to avoid very ugly overflow
            scale_x = min(scale_x, 1.5)
            scale_y = min(scale_y, 1.5)

            if scale_x > 1.001 or scale_y > 1.001:
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    poly = affinity.scale(poly, xfact=scale_x, yfact=scale_y, origin='center')
                    pts = np.array(poly.exterior.coords[:4])
                    dst_points = rotate_polygons(
                        region.center, pts.reshape(1, -1), -region.angle, to_int=False
                    ).reshape(-1, 4, 2).astype(np.int64)
                except Exception:
                    dst_points = region.min_rect
            else:
                dst_points = region.min_rect

        # --- 4. Store results ---------------------------------------------
        dst_points_list.append(dst_points)
        region.font_size = int(target_font_size)

    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    # If configuration is set to non-automatic mode, use configuration to determine direction directly
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    #print(f"Region text: {region.text}, forced_direction: {forced_direction}, render_horizontally: {render_horizontally}")

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
        )
    h, w, _ = temp_box.shape
    r_temp = w / h

    # Extend temporary box so that it has same ratio as original
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    if region.horizontal:  
        #print("Processing HORIZONTAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:  
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # Columns fully filled, rows centered
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()  
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) // 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [:, :w] = [0:{h}, 0:{w}]")  
         
                # The line is full, and there should be no empty columns on the left side of the text. Otherwise, when multiple text boxes are aligned on the left, the translated text cannot be aligned. Common scenarios: borderless comics, comic postscript.  
                # When there are bubbles on the current page, it can be changed to center: box[0:h, w_ext:w_ext+w] = temp_box, requiring more accurate bubble detection. But not changing it doesn't have much impact.
                box[0:h, 0:w] = temp_box  
            else:  
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()  
    else:  
        #print("Processing VERTICAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:   
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, 0:w] = [0:{h}, 0:{w}]")  
                # The rows are full, and there should be no empty lines above the text; otherwise, when multiple text boxes have their top edges aligned, the text cannot be aligned. Common scenario: borderless comics, CG. 
                # When there are bubbles on the current page, it can be changed to center: box[h_ext:h_ext+h, 0:w] = temp_box, requiring more accurate bubble detection.
                box[0:h, 0:w] = temp_box  
            else:   
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()   
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) / 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]") 
                # Rows are fully filled, columns are centered
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()   
    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

