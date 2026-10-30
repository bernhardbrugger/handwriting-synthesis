"""Read a TensorFlow 1.x checkpoint (the `model-17900` bundle) without TensorFlow.

The index file is a TensorFlow SSTable. Each tensor entry is a small protobuf
pointing at a byte range in the `.data-00000-of-00001` shard.
"""

import struct
from pathlib import Path

import numpy as np


_TABLE_MAGIC = 0xDB4775248B80FB57
_TABLE_FOOTER_LENGTH = 48
_BLOCK_TRAILER = 5


class CheckpointError(RuntimeError):
    pass


def _read_varint(buf, index):
    shift = 0
    result = 0
    while True:
        if index >= len(buf):
            raise CheckpointError('truncated varint in the checkpoint')
        byte = buf[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, index
        shift += 7
        if shift > 70:
            raise CheckpointError('varint in the checkpoint is too long')


def _parse_fields(buf):
    index = 0
    fields = []
    while index < len(buf):
        key, index = _read_varint(buf, index)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, index = _read_varint(buf, index)
        elif wire_type == 1:
            value = buf[index:index + 8]
            index += 8
        elif wire_type == 2:
            length, index = _read_varint(buf, index)
            value = buf[index:index + length]
            index += length
        elif wire_type == 5:
            value = buf[index:index + 4]
            index += 4
        else:
            raise CheckpointError('unknown protobuf type in the checkpoint')
        fields.append((field_number, wire_type, value))
    return fields


def _parse_block(data):
    if len(data) < 4:
        raise CheckpointError('checkpoint block is too short')
    num_restarts = struct.unpack_from('<I', data, len(data) - 4)[0]
    if num_restarts < 1 or (1 + num_restarts) * 4 > len(data):
        raise CheckpointError('checkpoint block has invalid restart points')
    end = len(data) - (1 + num_restarts) * 4
    entries = []
    index = 0
    key = b''
    while index < end:
        shared, index = _read_varint(data, index)
        unshared, index = _read_varint(data, index)
        value_length, index = _read_varint(data, index)
        if shared > len(key):
            raise CheckpointError('broken key in the checkpoint index')
        key = key[:shared] + data[index:index + unshared]
        index += unshared
        value = data[index:index + value_length]
        index += value_length
        entries.append((key, value))
    return entries


def _block_handle(buf, index):
    offset, index = _read_varint(buf, index)
    size, index = _read_varint(buf, index)
    return offset, size, index


def _load_block(raw, offset, size):
    block = raw[offset:offset + size]
    trailer = raw[offset + size:offset + size + _BLOCK_TRAILER]
    if len(block) != size or len(trailer) != _BLOCK_TRAILER:
        raise CheckpointError('checkpoint index is truncated')
    if trailer[0] != 0:
        raise CheckpointError('compressed checkpoint blocks are not supported')
    return block


def _tensor_entry(blob):
    dtype = None
    shape = []
    offset = 0
    size = None
    for field_number, wire_type, value in _parse_fields(blob):
        if field_number == 1 and wire_type == 0:
            dtype = value
        elif field_number == 2 and wire_type == 2:
            for dim_field, dim_wire, dim_value in _parse_fields(value):
                if dim_field == 2 and dim_wire == 2:
                    for size_field, size_wire, size_value in _parse_fields(dim_value):
                        if size_field == 1 and size_wire == 0:
                            shape.append(size_value)
        elif field_number == 4 and wire_type == 0:
            offset = value
        elif field_number == 5 and wire_type == 0:
            size = value
    return dtype, shape, offset, size


def _index_entries(index_path):
    raw = Path(index_path).read_bytes()
    if len(raw) < _TABLE_FOOTER_LENGTH:
        raise CheckpointError('checkpoint index is too small')
    magic = struct.unpack_from('<Q', raw, len(raw) - 8)[0]
    if magic != _TABLE_MAGIC:
        raise CheckpointError('checkpoint index has no TensorFlow signature')
    footer = raw[-_TABLE_FOOTER_LENGTH:]
    _, _, cursor = _block_handle(footer, 0)
    index_offset, index_size, _ = _block_handle(footer, cursor)
    index_block = _load_block(raw, index_offset, index_size)
    entries = []
    for _key, handle in _parse_block(index_block):
        offset, size, _ = _block_handle(handle, 0)
        entries.extend(_parse_block(_load_block(raw, offset, size)))
    return entries


def load_weights(prefix, names):
    """Load named float32 tensors from a checkpoint prefix.

    `prefix` is the path without extension, for example
    `checkpoints/model-17900`.
    """
    prefix = Path(prefix)
    index_path = prefix.parent / (prefix.name + '.index')
    data_path = prefix.parent / (prefix.name + '.data-00000-of-00001')
    if not index_path.is_file() or not data_path.is_file():
        raise CheckpointError(
            'checkpoint not found: {} (.index and .data-00000-of-00001)'.format(prefix)
        )

    wanted = set(names)
    located = {}
    for key, blob in _index_entries(index_path):
        name = key.decode('utf-8')
        if name not in wanted:
            continue
        dtype, shape, offset, size = _tensor_entry(blob)
        if dtype != 1:
            raise CheckpointError('{} is not a float32 tensor'.format(name))
        if size is None:
            raise CheckpointError('{} has no size in the index'.format(name))
        located[name] = (shape, offset, size)

    missing = sorted(wanted - set(located))
    if missing:
        raise CheckpointError('weights missing from the checkpoint: {}'.format(', '.join(missing)))

    return _read_tensors(data_path, located)


def _read_tensors(data_path, located):
    weights = {}
    with data_path.open('rb') as handle:
        for name, (shape, offset, size) in located.items():
            handle.seek(offset)
            raw = handle.read(size)
            if len(raw) != size:
                raise CheckpointError('weight data for {} is truncated'.format(name))
            count = 1
            for dim in shape:
                count *= dim
            if count * 4 != size:
                raise CheckpointError('size of {} does not match shape {}'.format(name, shape))
            array = np.frombuffer(raw, dtype='<f4')
            weights[name] = array.reshape(shape).copy() if shape else array.copy()
    return weights
