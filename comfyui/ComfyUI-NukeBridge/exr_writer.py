"""
Minimal multi-channel OpenEXR writer (scanline, ZIP compression), numpy + zlib only.
No OpenEXR / OpenCV dependency, so it runs in any ComfyUI install.
"""

import struct
import zlib

import numpy as np

HALF, FLOAT = 1, 2
_NP = {HALF: np.float16, FLOAT: np.float32}
_ZIP_LINES = 16


def _attr(name, typ, data):
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def _zip(raw):
    """OpenEXR ZIP: byte-interleave + delta predictor + zlib."""
    b = np.frombuffer(raw, dtype=np.uint8)
    t = np.concatenate([b[0::2], b[1::2]])
    d = t.astype(np.int16)
    d[1:] = (t[1:].astype(np.int16) - t[:-1].astype(np.int16) + 128 + 256) & 0xFF
    return zlib.compress(d.astype(np.uint8).tobytes(), 4)


def write_exr(path, channels, attributes=None, compress=True):
    """
    channels   : {"depth.Z": (array HxW, FLOAT), "N.R": (array, HALF), ...}
    attributes : {"name": float | str} extra header metadata
    """
    names = sorted(channels)                       # EXR requires alphabetical order
    h, w = channels[names[0]][0].shape
    data = []
    chlist = b""
    for n in names:
        arr, pt = channels[n]
        if arr.shape != (h, w):
            raise ValueError("channel %s has shape %s, expected %s" % (n, arr.shape, (h, w)))
        data.append(np.ascontiguousarray(np.asarray(arr).astype(np.dtype(_NP[pt]).newbyteorder("<"))))
        chlist += n.encode() + b"\0" + struct.pack("<iB3xii", pt, 0, 1, 1)
    chlist += b"\0"

    comp = 3 if compress else 0
    lines = _ZIP_LINES if compress else 1
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = struct.pack("<ii", 20000630, 2)
    header += _attr("channels", "chlist", chlist)
    header += _attr("compression", "compression", struct.pack("<B", comp))
    header += _attr("dataWindow", "box2i", box)
    header += _attr("displayWindow", "box2i", box)
    header += _attr("lineOrder", "lineOrder", struct.pack("<B", 0))
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    for k, v in (attributes or {}).items():
        if isinstance(v, str):
            b = v.encode()
            header += _attr(k, "string", b)
        else:
            header += _attr(k, "float", struct.pack("<f", float(v)))
    header += b"\0"

    chunks = []
    for y0 in range(0, h, lines):
        y1 = min(y0 + lines, h)
        raw = b"".join(arr[y].tobytes() for y in range(y0, y1) for arr in data)
        payload = raw
        if compress:
            z = _zip(raw)
            if len(z) < len(raw):
                payload = z
        chunks.append(struct.pack("<ii", y0, len(payload)) + payload)

    offset = len(header) + 8 * len(chunks)
    table = b""
    for c in chunks:
        table += struct.pack("<Q", offset)
        offset += len(c)
    with open(path, "wb") as f:
        f.write(header)
        f.write(table)
        for c in chunks:
            f.write(c)
