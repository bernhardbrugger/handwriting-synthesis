"""Character set of the pretrained model, plus a few ASCII rewrites."""

import unicodedata


# Same inventory as the original training code, including the null character
# that marks the end of a line. Upper-case Q, X and Z were not in the data.
ALPHABET = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
]

ALPHA_TO_NUM = {char: index for index, char in enumerate(ALPHABET)}
MAX_CHAR_LEN = 75

_UMLAUTS = (
    ('Ä', 'Ae'),
    ('Ö', 'Oe'),
    ('Ü', 'Ue'),
    ('ä', 'ae'),
    ('ö', 'oe'),
    ('ü', 'ue'),
    ('ẞ', 'SS'),
    ('ß', 'ss'),
)


class TextError(ValueError):
    pass


def transliterate(text):
    """Rewrite characters the model never saw into the training alphabet.

    German umlauts become ae, oe, ue and ss, with a capital first letter
    when the source letter is capital. Curly quotes and long dashes become
    plain ASCII.
    """
    text = unicodedata.normalize('NFC', text)
    text = (
        text.replace('\u00a0', ' ')
        .replace('\t', ' ')
        .replace('“', '"')
        .replace('”', '"')
        .replace('„', '"')
        .replace('‘', "'")
        .replace('’', "'")
        .replace('–', '-')
        .replace('—', '-')
        .replace('…', '...')
    )
    for source, target in _UMLAUTS:
        text = text.replace(source, target)
    return text


def validate_line(line):
    invalid = sorted({char for char in line if char not in ALPHA_TO_NUM})
    if not invalid:
        return
    shown = ', '.join('{} (U+{:04X})'.format(char, ord(char)) for char in invalid)
    raise TextError(
        'Unknown character: {}. Capital Q, X and Z are not in the alphabet. '
        'Digits and simple punctuation are allowed. Umlauts are rewritten '
        'to ae, oe, ue and ss beforehand.'.format(shown)
    )


def wrap_line(line, limit=MAX_CHAR_LEN):
    """Break a line so each piece fits the model's character window."""
    if line == '':
        return ['']
    pieces = []
    current = ''
    for word in line.split(' '):
        while len(word) > limit:
            if current:
                pieces.append(current)
                current = ''
            pieces.append(word[:limit])
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = current + ' ' + word
        else:
            pieces.append(current)
            current = word
    if current or not pieces:
        pieces.append(current)
    return pieces


def encode(line):
    """Map a validated line to model ids, with a trailing null."""
    return [ALPHA_TO_NUM[char] for char in line] + [0]
