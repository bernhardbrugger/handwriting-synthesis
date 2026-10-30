"""Public API: text in, SVG handwriting out."""

import re
from pathlib import Path

import numpy as np

from .alphabet import (
    MAX_CHAR_LEN,
    TextError,
    encode,
    transliterate,
    validate_line,
    wrap_line,
)
from .draw import render_svg
from .model import HandwritingModel, pad_chars


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIAS = 0.75
_COLOR_RE = re.compile(r'^(?:#[0-9A-Fa-f]{3}|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{8}|[A-Za-z]{1,32})$')
_MAX_LINES = 40
_MAX_CHARS = 2000


class HandwritingError(ValueError):
    pass


def _checkpoint_prefix(path):
    prefix = Path(path)
    index = prefix.parent / (prefix.name + '.index')
    data = prefix.parent / (prefix.name + '.data-00000-of-00001')
    return index.is_file() and data.is_file()


def find_checkpoint(explicit=None):
    if explicit is not None:
        prefix = Path(explicit)
        if not _checkpoint_prefix(prefix):
            raise HandwritingError('Checkpoint not found: {}'.format(prefix))
        return prefix
    for candidate in (ROOT / 'checkpoints' / 'model-17900', Path.cwd() / 'checkpoints' / 'model-17900'):
        if _checkpoint_prefix(candidate):
            return candidate
    raise HandwritingError(
        'checkpoints/model-17900 was not found next to the package or in the current directory.'
    )


def find_styles(explicit=None):
    if explicit is not None:
        directory = Path(explicit)
    else:
        directory = ROOT / 'styles'
        if not directory.is_dir():
            directory = Path.cwd() / 'styles'
    if not directory.is_dir():
        raise HandwritingError('Styles directory not found: {}'.format(directory))
    return directory


def list_styles(styles_dir=None):
    directory = find_styles(styles_dir)
    found = []
    for path in sorted(directory.glob('style-*-chars.npy')):
        stem = path.name[len('style-'):-len('-chars.npy')]
        strokes = directory / 'style-{}-strokes.npy'.format(stem)
        if stem.isdigit() and strokes.is_file():
            text = np.load(path).tobytes().decode('utf-8').replace('\x00', '')
            found.append((int(stem), text))
    return found


def _as_lines(text):
    if isinstance(text, str):
        lines = text.splitlines() or ['']
    else:
        lines = []
        for item in text:
            pieces = str(item).splitlines()
            lines.extend(pieces or [''])
    if not ''.join(lines).strip():
        raise HandwritingError('Text is missing.')
    if sum(len(line) for line in lines) > _MAX_CHARS:
        raise HandwritingError('The text is too long (at most {} characters).'.format(_MAX_CHARS))
    return lines


def _broadcast(value, count, default, name):
    if value is None:
        return [default] * count
    if isinstance(value, str) or not isinstance(value, (list, tuple, np.ndarray)):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise HandwritingError(
            '{} needs one value or exactly {} values, got {}.'.format(name, count, len(values))
        )
    return values


def _check_color(color):
    if not isinstance(color, str) or not _COLOR_RE.match(color):
        raise HandwritingError('Invalid color: {!r}'.format(color))
    return color


def _check_width(width):
    try:
        number = float(width)
    except (TypeError, ValueError):
        raise HandwritingError('Invalid stroke width: {!r}'.format(width))
    if not np.isfinite(number) or number <= 0 or number > 20:
        raise HandwritingError('Stroke width must be between 0 and 20.')
    return number


def _check_bias(bias):
    try:
        number = float(bias)
    except (TypeError, ValueError):
        raise HandwritingError('Invalid bias: {!r}'.format(bias))
    if not np.isfinite(number) or number < -1 or number > 5:
        raise HandwritingError('Bias must be between -1 and 5.')
    return number


class Hand(object):
    """Samples handwriting from checkpoints/model-17900 and writes SVG."""

    def __init__(self, checkpoint=None, styles_dir=None):
        self.checkpoint = find_checkpoint(checkpoint)
        self.styles_dir = find_styles(styles_dir)
        self.model = HandwritingModel.from_checkpoint(self.checkpoint)
        self._style_cache = {}

    def _style(self, style_id):
        try:
            style_id = int(style_id)
        except (TypeError, ValueError):
            raise HandwritingError('Invalid style: {!r}'.format(style_id))
        if style_id not in self._style_cache:
            chars_path = self.styles_dir / 'style-{}-chars.npy'.format(style_id)
            strokes_path = self.styles_dir / 'style-{}-strokes.npy'.format(style_id)
            if not chars_path.is_file() or not strokes_path.is_file():
                known = ', '.join(str(index) for index, _text in list_styles(self.styles_dir))
                raise HandwritingError('Style {} does not exist. Available: {}.'.format(style_id, known))
            text = np.load(chars_path).tobytes().decode('utf-8').replace('\x00', '')
            strokes = np.load(strokes_path).astype(np.float32)
            self._style_cache[style_id] = (text, strokes)
        return self._style_cache[style_id]

    def prepare(self, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        """Transliterate, validate and wrap lines. Attributes follow each fragment."""
        source = _as_lines(lines)
        count = len(source)
        bias_values = [_check_bias(value) for value in _broadcast(biases, count, DEFAULT_BIAS, 'Bias')]
        color_values = [_check_color(value) for value in _broadcast(stroke_colors, count, 'black', 'color')]
        width_values = [_check_width(value) for value in _broadcast(stroke_widths, count, 2, 'stroke width')]
        if styles is None:
            style_values = [None] * count
        else:
            style_values = _broadcast(styles, count, None, 'style')

        prepared = []
        for line, bias, style, color, width in zip(
            source, bias_values, style_values, color_values, width_values
        ):
            rewritten = transliterate(line)
            try:
                validate_line(rewritten)
            except TextError as exc:
                raise HandwritingError(str(exc))
            for piece in wrap_line(rewritten, MAX_CHAR_LEN):
                prepared.append({
                    'line': piece,
                    'bias': bias,
                    'style': None if style is None else style,
                    'color': color,
                    'width': width,
                })
        if len(prepared) > _MAX_LINES:
            raise HandwritingError(
                'Too many lines after wrapping (at most {}).'.format(_MAX_LINES)
            )
        return prepared

    def _sample_prepared(self, prepared, rng):
        jobs = [item for item in prepared if item['line']]
        strokes = {id(item): None for item in prepared}
        if not jobs:
            return [None] * len(prepared)

        encoded = []
        primes = []
        prime_lengths = []
        for item in jobs:
            if item['style'] is None:
                encoded.append(encode(item['line']))
                primes.append(None)
                prime_lengths.append(0)
            else:
                style_text, style_strokes = self._style(item['style'])
                combined = style_text + ' ' + item['line']
                try:
                    validate_line(combined)
                except TextError as exc:
                    raise HandwritingError(str(exc))
                encoded.append(encode(combined))
                primes.append(style_strokes)
                prime_lengths.append(len(style_strokes))

        chars, lengths = pad_chars(encoded)
        bias = np.array([item['bias'] for item in jobs], dtype=np.float32)
        max_steps = 40 * max(len(item['line']) for item in jobs)
        if any(prime is not None for prime in primes):
            max_prime = max(prime_lengths)
            prime = np.zeros((len(jobs), max_prime, 3), dtype=np.float32)
            for index, stroke in enumerate(primes):
                if stroke is not None and len(stroke):
                    prime[index, :len(stroke)] = stroke
            prime_length_array = np.array(prime_lengths, dtype=np.int32)
        else:
            prime = None
            prime_length_array = None

        samples = self.model.sample(
            chars,
            lengths,
            bias,
            rng,
            max_steps=max_steps,
            prime=prime,
            prime_lengths=prime_length_array,
        )
        for item, sample in zip(jobs, samples):
            strokes[id(item)] = sample
        return [strokes[id(item)] for item in prepared]

    def svg(self, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, seed=None):
        prepared = self.prepare(lines, biases, styles, stroke_colors, stroke_widths)
        rng = np.random.default_rng(seed)
        samples = self._sample_prepared(prepared, rng)
        return render_svg(
            samples,
            [item['color'] for item in prepared],
            [item['width'] for item in prepared],
        ), prepared

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, seed=None):
        document, prepared = self.svg(
            lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths,
            seed=seed,
        )
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding='utf-8')
        return prepared
