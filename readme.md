![](img/banner.svg)
# Handwriting Synthesis

Inference for the handwriting synthesis experiments in the paper [Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) by Alex Graves. The network is the pretrained checkpoint `checkpoints/model-17900`. Sampling runs in NumPy and SciPy, without TensorFlow.

## Install

```bash
pip install -r requirements.txt
```

That installs NumPy and SciPy only.

## Usage

Write one SVG page:

```bash
python -m handwriting "Now this is a story all about how" -o page.svg
```

Several lines, a calmer hand, and one of the stored styles (0–12):

```bash
python -m handwriting "Line one" "Line two" -o page.svg --bias 0.75 --style 9 --color navy
```

`--seed 1` repeats the same page. `-o -` prints the SVG to stdout. The training alphabet has no capital Q, X or Z. ä, ö, ü and ß are spelled ae, oe, ue and ss, so those words still draw.

A local page, bound to this machine:

```bash
python -m handwriting.server
```

Open http://127.0.0.1:8000/ . The form takes text, bias, style, color and stroke width.

```python
from handwriting import Hand

lines = [
    "Now this is a story all about how",
    "My life got flipped turned upside down",
    "And I'd like to take a minute, just sit right there",
    "I'll tell you how I became the prince of a town called Bel-Air",
]
hand = Hand()
hand.write(
    filename='img/usage_demo.svg',
    lines=lines,
    biases=[.75 for _ in lines],
    styles=[9 for _ in lines],
    stroke_colors=['red', 'green', 'black', 'blue'],
    stroke_widths=[1, 2, 1, 2],
)
```

![](img/usage_demo.svg)

Higher bias makes the samples neater. A style primes the network with a real line from `styles/` and then writes the new text in that hand.

Training is not included. The weights in `checkpoints/model-17900` are the ones used for the pictures below.

## Demonstrations

### Demo #1

Fixed style and fixed bias.

**Smash Mouth – All Star ([lyrics](https://www.azlyrics.com/lyrics/smashmouth/allstar.html))**
![](img/all_star.svg)

### Demo #2

Fixed bias, a different style for each verse.

**Vanessa Carlton – A Thousand Miles ([lyrics](https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html))**
![](img/downtown.svg)

### Demo #3

Fixed style, bias lowered on each verse. The last verse is unbiased.

**Leonard Cohen – Hallelujah ([lyrics](https://www.youtube.com/watch?v=dQw4w9WgXcQ))**
![](img/give_up.svg)
