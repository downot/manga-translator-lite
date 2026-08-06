def is_tiny_component_inside_line(
    area: int, overlap_ratio: float, keep_threshold: float
) -> bool:
    """Keep small detector components only when they are inside a text line."""
    return area <= 9 and overlap_ratio > keep_threshold


_is_tiny_component_inside_line = is_tiny_component_inside_line
