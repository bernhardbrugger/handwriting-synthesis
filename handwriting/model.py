"""NumPy inference for the Graves handwriting network.

Three LSTMs with a monotonic window over the character sequence, then a
mixture of bivariate Gaussians for the pen offset and a Bernoulli for the
end of a stroke. Gate order and the forget bias follow the TensorFlow 1.6
LSTMCell that produced checkpoints/model-17900: input, cell, forget, output,
with forget bias 1.
"""

from pathlib import Path

import numpy as np

from .alphabet import ALPHABET
from .checkpoint import CheckpointError, load_weights


_WEIGHT_NAMES = (
    'rnn/LSTMAttentionCell/lstm_cell/kernel',
    'rnn/LSTMAttentionCell/lstm_cell/bias',
    'rnn/LSTMAttentionCell/lstm_cell_1/kernel',
    'rnn/LSTMAttentionCell/lstm_cell_1/bias',
    'rnn/LSTMAttentionCell/lstm_cell_2/kernel',
    'rnn/LSTMAttentionCell/lstm_cell_2/bias',
    'rnn/LSTMAttentionCell/attention/weights',
    'rnn/LSTMAttentionCell/attention/biases',
    'rnn/gmm/weights',
    'rnn/gmm/biases',
)

# The original sampler pads every character string to this width, so the
# attention peak is allowed to move past the real text and end the line.
_CHAR_BUFFER = 120
_FORGET_BIAS = np.float32(1.0)
_KAPPA_SCALE = np.float32(25.0)


class _State(object):
    __slots__ = ('h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'kappa', 'w', 'phi')


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _softplus(values):
    return np.logaddexp(np.float32(0.0), values)


def _softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _lstm(inputs, hidden, cell, kernel, bias):
    gates = np.concatenate([inputs, hidden], axis=1) @ kernel
    gates = gates + bias
    gate_input, candidate, forget, output = np.split(gates, 4, axis=1)
    new_cell = _sigmoid(forget + _FORGET_BIAS) * cell + _sigmoid(gate_input) * np.tanh(candidate)
    new_hidden = np.tanh(new_cell) * _sigmoid(output)
    return new_hidden.astype(np.float32), new_cell.astype(np.float32)


def _mix(active, new, old):
    mask = active[:, None]
    mixed = _State()
    for name in _State.__slots__:
        setattr(mixed, name, np.where(mask, getattr(new, name), getattr(old, name)))
    return mixed


class HandwritingModel(object):
    def __init__(self, weights):
        self.lstm1_kernel = weights['rnn/LSTMAttentionCell/lstm_cell/kernel']
        self.lstm1_bias = weights['rnn/LSTMAttentionCell/lstm_cell/bias']
        self.lstm2_kernel = weights['rnn/LSTMAttentionCell/lstm_cell_1/kernel']
        self.lstm2_bias = weights['rnn/LSTMAttentionCell/lstm_cell_1/bias']
        self.lstm3_kernel = weights['rnn/LSTMAttentionCell/lstm_cell_2/kernel']
        self.lstm3_bias = weights['rnn/LSTMAttentionCell/lstm_cell_2/bias']
        self.attn_w = weights['rnn/LSTMAttentionCell/attention/weights']
        self.attn_b = weights['rnn/LSTMAttentionCell/attention/biases']
        self.gmm_w = weights['rnn/gmm/weights']
        self.gmm_b = weights['rnn/gmm/biases']

        self.lstm_size = int(self.lstm1_bias.shape[0] // 4)
        self.n_attn = int(self.attn_b.shape[0] // 3)
        self.alphabet = int(self.attn_w.shape[0] - 3 - self.lstm_size)
        self.n_components = int((self.gmm_b.shape[0] - 1) // 6)
        self._check_shapes()

    def _check_shapes(self):
        hidden = self.lstm_size
        letters = self.alphabet
        components = self.n_components
        expected = {
            'erstes LSTM': self.lstm1_kernel.shape == (letters + 3 + hidden, 4 * hidden),
            'zweites LSTM': self.lstm2_kernel.shape == (3 + hidden + letters + hidden, 4 * hidden),
            'drittes LSTM': self.lstm3_kernel.shape == self.lstm2_kernel.shape,
            'Aufmerksamkeit': self.attn_w.shape == (letters + 3 + hidden, 3 * self.n_attn),
            'Mischung': self.gmm_w.shape == (hidden, 6 * components + 1),
        }
        bad = [name for name, ok in expected.items() if not ok]
        if bad or letters != len(ALPHABET):
            raise CheckpointError(
                'Checkpoint passt nicht zum Handschrift-Netz ({}).'.format(
                    ', '.join(bad) or 'Alphabet'
                )
            )

    @classmethod
    def from_checkpoint(cls, prefix):
        return cls(load_weights(Path(prefix), _WEIGHT_NAMES))

    def zero_state(self, batch, char_len):
        def zeros(width):
            return np.zeros((batch, width), dtype=np.float32)

        state = _State()
        state.h1 = zeros(self.lstm_size)
        state.c1 = zeros(self.lstm_size)
        state.h2 = zeros(self.lstm_size)
        state.c2 = zeros(self.lstm_size)
        state.h3 = zeros(self.lstm_size)
        state.c3 = zeros(self.lstm_size)
        state.kappa = zeros(self.n_attn)
        state.w = zeros(self.alphabet)
        state.phi = np.zeros((batch, char_len), dtype=np.float32)
        return state

    def step(self, state, inputs, one_hot, mask):
        hidden1, cell1 = _lstm(
            np.concatenate([state.w, inputs], axis=1),
            state.h1,
            state.c1,
            self.lstm1_kernel,
            self.lstm1_bias,
        )
        attention_in = np.concatenate([state.w, inputs, hidden1], axis=1)
        attention = _softplus(attention_in @ self.attn_w + self.attn_b)
        alpha, beta, kappa_hat = np.split(attention, 3, axis=1)
        kappa = state.kappa + kappa_hat / _KAPPA_SCALE
        beta = np.maximum(beta, np.float32(0.01))

        positions = np.arange(one_hot.shape[1], dtype=np.float32)
        delta = kappa[:, :, None] - positions[None, None, :]
        window = alpha[:, :, None] * np.exp(-(delta * delta) / beta[:, :, None])
        phi = window.sum(axis=1).astype(np.float32)
        weights = ((phi * mask)[:, None, :] @ one_hot).squeeze(1).astype(np.float32)

        hidden2, cell2 = _lstm(
            np.concatenate([inputs, hidden1, weights], axis=1),
            state.h2,
            state.c2,
            self.lstm2_kernel,
            self.lstm2_bias,
        )
        hidden3, cell3 = _lstm(
            np.concatenate([inputs, hidden2, weights], axis=1),
            state.h3,
            state.c3,
            self.lstm3_kernel,
            self.lstm3_bias,
        )

        new_state = _State()
        new_state.h1 = hidden1
        new_state.c1 = cell1
        new_state.h2 = hidden2
        new_state.c2 = cell2
        new_state.h3 = hidden3
        new_state.c3 = cell3
        new_state.kappa = kappa.astype(np.float32)
        new_state.w = weights
        new_state.phi = phi
        return new_state

    def _mixture(self, hidden, bias):
        raw = hidden @ self.gmm_w + self.gmm_b
        count = self.n_components
        pis = raw[:, :count]
        sigmas = raw[:, count:3 * count]
        rhos = raw[:, 3 * count:4 * count]
        mus = raw[:, 4 * count:6 * count]
        eos = raw[:, 6 * count:]
        bias = np.asarray(bias, dtype=np.float32).reshape(-1, 1)
        pis = _softmax(pis * (1.0 + bias))
        pis = np.where(pis < 0.01, np.float32(0.0), pis)
        sigmas = np.exp(np.clip(sigmas - bias, -20.0, 20.0))
        sigmas = np.maximum(sigmas, np.float32(1e-4))
        rhos = np.clip(np.tanh(rhos), 1e-8 - 1.0, 1.0 - 1e-8)
        eos = np.clip(_sigmoid(eos), 1e-8, 1.0 - 1e-8)
        eos = np.where(eos < 0.01, np.float32(0.0), eos)
        return (
            pis.astype(np.float32),
            mus[:, :count],
            mus[:, count:],
            sigmas[:, :count],
            sigmas[:, count:],
            rhos.astype(np.float32),
            eos.astype(np.float32),
        )

    def _draw(self, hidden, bias, rng):
        pis, mu1, mu2, sigma1, sigma2, rho, eos = self._mixture(hidden, bias)
        totals = pis.sum(axis=1, keepdims=True)
        uniform = np.full_like(pis, 1.0 / pis.shape[1])
        probabilities = np.where(totals > 0, pis / np.maximum(totals, 1e-8), uniform)
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        choice = (cumulative < rng.random(len(pis))[:, None]).sum(axis=1).astype(np.int32)
        row = np.arange(len(pis))
        chosen_mu1 = mu1[row, choice]
        chosen_mu2 = mu2[row, choice]
        chosen_s1 = sigma1[row, choice]
        chosen_s2 = sigma2[row, choice]
        chosen_rho = rho[row, choice]
        noise1 = rng.standard_normal(len(pis)).astype(np.float32)
        noise2 = rng.standard_normal(len(pis)).astype(np.float32)
        scale = np.sqrt(np.maximum(1.0 - chosen_rho * chosen_rho, 0.0)).astype(np.float32)
        x = chosen_mu1 + chosen_s1 * noise1
        y = chosen_mu2 + chosen_s2 * (chosen_rho * noise1 + scale * noise2)
        pen_up = (rng.random(len(pis)) < eos[:, 0]).astype(np.float32)
        return np.stack([x, y, pen_up], axis=1).astype(np.float32)

    def _finished(self, state, lengths, bias, rng):
        # A separate draw decides whether the pen lifted. The point that is
        # actually written is sampled afterwards, as in the original graph.
        char_idx = np.argmax(state.phi, axis=1)
        on_last = char_idx >= (lengths - 1)
        past_end = char_idx >= lengths
        pen_up = self._draw(state.h3, bias, rng)[:, 2] > 0.5
        return (on_last & pen_up) | past_end

    def _teacher_force(self, state, strokes, stroke_lengths, one_hot, mask):
        for step in range(strokes.shape[1]):
            active = stroke_lengths > step
            if not np.any(active):
                break
            updated = self.step(state, strokes[:, step, :], one_hot, mask)
            state = _mix(active, updated, state)
        return state

    def sample(self, chars, lengths, bias, rng, max_steps, prime=None, prime_lengths=None):
        """Sample pen offsets for a batch of character rows.

        `chars` is int32 [batch, time] and already contains the trailing null.
        Returns a list of float32 arrays with shape [steps, 3].
        """
        batch, char_len = chars.shape
        one_hot = np.eye(self.alphabet, dtype=np.float32)[chars]
        positions = np.arange(char_len)[None, :]
        mask = (positions < lengths[:, None]).astype(np.float32)
        state = self.zero_state(batch, char_len)
        bias = np.asarray(bias, dtype=np.float32).reshape(batch)
        inputs = np.zeros((batch, 3), dtype=np.float32)
        inputs[:, 2] = 1.0
        alive = np.ones(batch, dtype=bool)

        if prime is not None:
            state = self._teacher_force(state, prime, prime_lengths, one_hot, mask)
            primed = np.asarray(prime_lengths) > 0
            # The priming stroke is fed back once and is not part of the page.
            # Lines without a style still start from a pen-down at the origin.
            alive = alive & ~(primed & self._finished(state, lengths, bias, rng))
            inputs = np.where(primed[:, None], self._draw(state.h3, bias, rng), inputs)

        frames = []
        for _step in range(int(max_steps)):
            if not np.any(alive):
                break
            state = self.step(state, inputs, one_hot, mask)
            stop = self._finished(state, lengths, bias, rng)
            point = self._draw(state.h3, bias, rng)
            point = np.where(alive[:, None], point, np.float32(0.0))
            frames.append(point.astype(np.float32))
            alive = alive & ~stop
            inputs = np.where(alive[:, None], point, np.float32(0.0)).astype(np.float32)

        if not frames:
            empty = np.zeros((0, 3), dtype=np.float32)
            return [empty.copy() for _ in range(batch)]

        sequence = np.stack(frames, axis=1)
        samples = []
        for row in sequence:
            kept = row[~np.all(row == 0.0, axis=1)]
            samples.append(np.ascontiguousarray(kept, dtype=np.float32))
        return samples


def pad_chars(encoded_rows):
    """Pad encoded lines to the sampler buffer. Returns ids and lengths."""
    width = max(_CHAR_BUFFER, max(len(row) for row in encoded_rows))
    chars = np.zeros((len(encoded_rows), width), dtype=np.int32)
    lengths = np.zeros(len(encoded_rows), dtype=np.int32)
    for index, row in enumerate(encoded_rows):
        chars[index, :len(row)] = row
        lengths[index] = len(row)
    return chars, lengths
