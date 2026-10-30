"""Command line: python -m handwriting \"Text\" -o page.svg"""

import argparse
import sys

from .hand import DEFAULT_BIAS, Hand, HandwritingError


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m handwriting',
        description='Draw text with the handwriting model and save an SVG page.',
    )
    parser.add_argument(
        'text',
        nargs='+',
        help='Text. Extra arguments, or newlines, become separate lines.',
    )
    parser.add_argument('-o', '--output', required=True, help='SVG file, or - for standard output')
    parser.add_argument(
        '--bias',
        type=float,
        default=DEFAULT_BIAS,
        help='Bias. Higher is neater (default %(default)s)',
    )
    parser.add_argument(
        '--style',
        type=int,
        default=None,
        help='Style number from styles/. Omit for no priming.',
    )
    parser.add_argument('--color', default='black', help='Stroke color (default black)')
    parser.add_argument('--width', type=float, default=2, help='Stroke width (default %(default)s)')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed, so the same page can be drawn again',
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    text = '\n'.join(args.text)
    try:
        hand = Hand()
        document, _prepared = hand.svg(
            text,
            biases=args.bias,
            styles=args.style,
            stroke_colors=args.color,
            stroke_widths=args.width,
            seed=args.seed,
        )
    except HandwritingError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.output == '-':
        sys.stdout.write(document)
    else:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding='utf-8')
        print(path, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
