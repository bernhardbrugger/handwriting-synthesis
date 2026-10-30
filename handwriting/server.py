"""Local page: python -m handwriting.server"""

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .hand import DEFAULT_BIAS, Hand, HandwritingError, list_styles


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handwriting</title>
<style>
  :root {{ color-scheme: light; }}
  body {{
    margin: 0;
    background: #efeae0;
    color: #241f18;
    font-family: Palatino, "Iowan Old Style", Georgia, serif;
  }}
  main {{
    max-width: 920px;
    margin: 0 auto;
    padding: 28px 18px 56px;
  }}
  h1 {{
    font-weight: 500;
    font-size: 2.1rem;
    margin: 0 0 6px;
  }}
  p.lead {{
    margin: 0 0 18px;
    line-height: 1.45;
  }}
  form, .sheet {{
    background: #fff;
    border: 1px solid #e0d8cc;
    border-radius: 14px;
    padding: 16px 16px 18px;
  }}
  label {{
    display: block;
    margin: 14px 0 6px;
  }}
  textarea, input, select, button {{
    font: inherit;
    color: inherit;
  }}
  textarea {{
    width: 100%;
    box-sizing: border-box;
    min-height: 140px;
    padding: 10px;
    border: 1px solid #d9d0c3;
    border-radius: 8px;
    background: #fffdf9;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
  }}
  input[type="number"], input[type="text"], select {{
    width: 100%;
    box-sizing: border-box;
    padding: 8px;
    border: 1px solid #d9d0c3;
    border-radius: 8px;
    background: #fffdf9;
  }}
  input[type="range"] {{ width: 100%; }}
  button {{
    margin-top: 16px;
    background: #2c261e;
    color: #f7f3ea;
    border: 0;
    border-radius: 8px;
    padding: 10px 18px;
    cursor: pointer;
  }}
  .error {{
    background: #f8e8e4;
    border: 1px solid #e3b2a8;
    color: #6e241c;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 0 0 14px;
  }}
  .sheet {{ margin-top: 18px; }}
  .sheet svg {{ width: 100%; height: auto; display: block; }}
  .meta {{ color: #5e564c; font-size: 0.95rem; margin: 8px 0 0; }}
  a {{ color: #2c261e; }}
  @media (max-width: 700px) {{
    .grid {{ grid-template-columns: 1fr; }}
    h1 {{ font-size: 1.7rem; }}
  }}
</style>
</head>
<body>
<main>
  <h1>Handwriting</h1>
  <p class="lead">The model draws the text in ink. Higher bias is neater.
  Capital Q, X and Z are not in the alphabet.</p>
  {error}
  <form method="post" action="/">
    <label for="text">Text</label>
    <textarea id="text" name="text" required>{text}</textarea>
    <div class="grid">
      <div>
        <label for="bias">Bias <output id="bias-out">{bias}</output></label>
        <input id="bias" name="bias" type="range" min="0" max="2" step="0.05" value="{bias}">
      </div>
      <div>
        <label for="style">Style</label>
        <select id="style" name="style">{styles}</select>
      </div>
      <div>
        <label for="color">Color</label>
        <input id="color" name="color" type="color" value="{color}">
      </div>
      <div>
        <label for="width">Stroke width</label>
        <input id="width" name="width" type="number" min="0.5" max="8" step="0.5" value="{width}">
      </div>
      <div>
        <label for="seed">Seed, empty for a new page each time</label>
        <input id="seed" name="seed" type="text" inputmode="numeric" value="{seed}">
      </div>
    </div>
    <button type="submit">Write</button>
  </form>
  {result}
</main>
<script>
  var bias = document.getElementById('bias');
  var out = document.getElementById('bias-out');
  bias.addEventListener('input', function () {{ out.value = bias.value; }});
</script>
</body>
</html>
"""


def _style_options(selected):
    options = ['<option value="">no style</option>']
    for index, phrase in list_styles():
        mark = ' selected' if selected == str(index) else ''
        label = html.escape('{} — {}'.format(index, phrase))
        options.append('<option value="{}"{}>{}</option>'.format(index, mark, label))
    return '\n'.join(options)


def _page(text, bias, style, color, width, seed, error='', result=''):
    error_html = '<p class="error">{}</p>'.format(html.escape(error)) if error else ''
    return _PAGE.format(
        error=error_html,
        text=html.escape(text),
        bias=html.escape(str(bias)),
        styles=_style_options(style),
        color=html.escape(color),
        width=html.escape(str(width)),
        seed=html.escape(seed),
        result=result,
    )


def _field(form, name, default=''):
    values = form.get(name)
    if not values:
        return default
    return values[0]


class _Handler(BaseHTTPRequestHandler):
    hand = None

    def log_message(self, fmt, *args):
        print('[{}] {}'.format(self.log_date_time_string(), fmt % args))

    def _send(self, status, body, content_type='text/html; charset=utf-8'):
        payload = body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split('?', 1)[0] not in ('/', '/index.html'):
            self._send(404, _page('', DEFAULT_BIAS, '', '#1a1a1a', 2, '', error='This page does not exist.'))
            return
        self._send(200, _page('', DEFAULT_BIAS, '', '#1a1a1a', 2, ''))

    def do_POST(self):
        if self.path.split('?', 1)[0] != '/':
            self._send(404, _page('', DEFAULT_BIAS, '', '#1a1a1a', 2, '', error='This page does not exist.'))
            return
        length = int(self.headers.get('Content-Length', '0') or 0)
        if length > 100000:
            self._send(400, _page('', DEFAULT_BIAS, '', '#1a1a1a', 2, '', error='The request is too large.'))
            return
        raw = self.rfile.read(length).decode('utf-8', errors='replace')
        form = parse_qs(raw, keep_blank_values=True)
        text = _field(form, 'text')
        bias = _field(form, 'bias', str(DEFAULT_BIAS))
        style = _field(form, 'style').strip()
        color = _field(form, 'color', '#1a1a1a')
        width = _field(form, 'width', '2')
        seed_text = _field(form, 'seed').strip()
        try:
            seed = None if seed_text == '' else int(seed_text)
            style_value = None if style == '' else int(style)
            document, prepared = self.hand.svg(
                text,
                biases=float(bias),
                styles=style_value,
                stroke_colors=color,
                stroke_widths=float(width),
                seed=seed,
            )
        except (HandwritingError, ValueError) as exc:
            self._send(400, _page(text, bias, style, color, width, seed_text, error=str(exc)))
            return

        written = '\n'.join(item['line'] for item in prepared)
        embedded = document.split('\n', 1)[1] if document.startswith('<?xml') else document
        result = (
            '<section class="sheet">'
            + embedded
            + '<p class="meta">Written as: '
            + html.escape(written)
            + '</p></section>'
        )
        self._send(200, _page(text, bias, style, color, width, seed_text, result=result))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog='python -m handwriting.server',
        description='Local page that draws text as handwriting.',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Address (default 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default 8000)')
    args = parser.parse_args(argv)

    print('Loading checkpoint weights...')
    _Handler.hand = Hand()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = 'http://{}:{}/'.format(args.host, args.port)
    print('Ready at {}'.format(url))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
