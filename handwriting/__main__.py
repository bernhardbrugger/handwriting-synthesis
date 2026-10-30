"""Command line: python -m handwriting \"Text\" -o seite.svg"""

import argparse
import sys

from .hand import DEFAULT_BIAS, Hand, HandwritingError


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m handwriting',
        description='Schreibt Text mit dem Handschrift-Modell nach SVG. Umlaute werden zu ae, oe, ue und ss.',
    )
    parser.add_argument('text', nargs='+', help='Text. Mehrere Argumente oder Zeilenumbrüche werden zu Zeilen.')
    parser.add_argument('-o', '--output', required=True, help='SVG-Datei, oder - für die Standardausgabe')
    parser.add_argument(
        '--bias',
        type=float,
        default=DEFAULT_BIAS,
        help='Sauberkeit, höher ist ruhiger (Standard %(default)s)',
    )
    parser.add_argument('--style', type=int, default=None, help='Stilnummer aus styles/, sonst ohne Vorbild')
    parser.add_argument('--color', default='black', help='Strichfarbe (Standard black)')
    parser.add_argument('--width', type=float, default=2, help='Strichstärke (Standard %(default)s)')
    parser.add_argument('--seed', type=int, default=None, help='Zufallsstart, damit dieselbe Seite wieder entsteht')
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
