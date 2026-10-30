"""Turn pen offsets into an SVG page. Depends only on NumPy and SciPy."""

import numpy as np
from scipy.signal import savgol_filter


_LINE_HEIGHT = 60
_VIEW_WIDTH = 1000


def offsets_to_coords(offsets):
    if len(offsets) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xy = np.cumsum(offsets[:, :2], axis=0)
    return np.concatenate([xy, offsets[:, 2:3]], axis=1)


def align(coords):
    """Remove the global slant of a line, as in the original drawing code."""
    coords = np.array(coords, dtype=np.float64, copy=True)
    if len(coords) < 2:
        return coords
    x = coords[:, 0]
    y = coords[:, 1]
    design = np.column_stack([np.ones(len(x)), x])
    solution, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    offset, slope = solution
    if not np.isfinite(offset) or not np.isfinite(slope):
        return coords
    theta = np.arctan(slope)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    coords[:, :2] = coords[:, :2] @ rotation - offset
    return coords


def denoise(coords):
    """Smooth each stroke. Strokes shorter than the filter window stay as they are."""
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    cuts = np.where(coords[:, 2] == 1)[0] + 1
    strokes = np.split(coords, cuts, axis=0)
    pieces = []
    for stroke in strokes:
        if len(stroke) == 0:
            continue
        if len(stroke) >= 7:
            smoothed_x = savgol_filter(stroke[:, 0], 7, 3, mode='nearest')
            smoothed_y = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
            xy = np.column_stack([smoothed_x, smoothed_y])
            stroke = np.column_stack([xy, stroke[:, 2]])
        pieces.append(stroke)
    if not pieces:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pieces)


def _format_coord(value):
    text = '{:.2f}'.format(float(value))
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    if text == '-0':
        return '0'
    return text


def _path(coords):
    commands = []
    previous_end = 1.0
    for x, y, end_of_stroke in coords:
        command = 'M' if previous_end == 1.0 else 'L'
        commands.append('{}{} {}'.format(command, _format_coord(x), _format_coord(y)))
        previous_end = end_of_stroke
    return ' '.join(commands)


def render_svg(offset_lines, colors, widths):
    """Place each sampled line on a white page and return an SVG document."""
    if not (len(offset_lines) == len(colors) == len(widths)):
        raise ValueError('strokes, colors and widths must be the same length')

    view_height = _LINE_HEIGHT * (len(offset_lines) + 1)
    placed = []
    cursor = np.array([0.0, -(3.0 * _LINE_HEIGHT / 4.0)])
    for offsets in offset_lines:
        if offsets is None or len(offsets) == 0:
            placed.append(None)
            cursor = cursor + np.array([0.0, -_LINE_HEIGHT])
            continue
        scaled = np.array(offsets, dtype=np.float64, copy=True)
        scaled[:, :2] *= 1.5
        strokes = offsets_to_coords(scaled)
        strokes = denoise(strokes)
        if len(strokes) == 0:
            placed.append(None)
            cursor = cursor + np.array([0.0, -_LINE_HEIGHT])
            continue
        strokes[:, :2] = align(strokes[:, :2])
        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + cursor
        strokes[:, 0] += (_VIEW_WIDTH - strokes[:, 0].max()) / 2
        placed.append(strokes)
        cursor = cursor + np.array([0.0, -_LINE_HEIGHT])

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 {} {}" width="{}" height="{}">'
        ).format(_VIEW_WIDTH, view_height, _VIEW_WIDTH, view_height),
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for strokes, color, width in zip(placed, colors, widths):
        if strokes is None:
            continue
        elements.append(
            '<path d="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round"/>'.format(
                _path(strokes), color, _format_coord(width)
            )
        )
    elements.append('</svg>')
    return '\n'.join(elements) + '\n'
