"""Read-only safetensors row access for PLE tables too large for RAM.

No returned object borrows the mapping. Callers own only gathered row bytes,
so release() can discard resident pages even while those rows are in use.
The mapping is never registered as a Torch parameter or buffer.
"""

import json
import mmap
import operator
import struct
from pathlib import Path


class MappedPLETable:
    def __init__(self, path, prefix, *, scale_path=None, max_gather_rows=65536):
        if type(max_gather_rows) is not int or max_gather_rows <= 0:
            raise ValueError("max_gather_rows must be a positive integer")
        self.max_gather_rows = max_gather_rows
        self.path = Path(path)
        self._scale_table = None
        if scale_path is not None and Path(scale_path).resolve() != self.path.resolve():
            self._init_separate(Path(scale_path), prefix)
            return
        self._mapping = None
        with self.path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 128 * 1024 * 1024:
                raise ValueError("safetensors header exceeds bound")
            header = json.loads(stream.read(length))
            weight = header[prefix + ".weight"]
            scale = header[prefix + ".scale"]
            if weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0":
                raise ValueError("PLE requires FP8 E4M3 values and E8M0 scales")
            self.rows, self.width = weight["shape"]
            if self.rows <= 0 or self.width <= 0 or self.width % 32:
                raise ValueError("invalid block-32 PLE shape")
            if scale["shape"] != [self.rows, self.width // 32]:
                raise ValueError("PLE scale shape mismatch")
            self._regions = []
            file_size = self.path.stat().st_size
            for item, row_bytes in ((weight, self.width), (scale, self.width // 32)):
                start, end = item["data_offsets"]
                if (start < 0 or end - start != self.rows * row_bytes
                        or 8 + length + end > file_size):
                    raise ValueError("invalid PLE payload extent")
                self._regions.append((8 + length + start, row_bytes))
            w_start, w_size = self._regions[0]
            s_start, s_size = self._regions[1]
            if not (w_start + self.rows * w_size <= s_start
                    or s_start + self.rows * s_size <= w_start):
                raise ValueError("overlapping PLE payloads")
            self._mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._mapping.madvise(mmap.MADV_RANDOM)

    def _init_separate(self, scale_path, prefix):
        """Map independently sharded complete tensors without concatenation."""
        self._mapping = None
        mappings, regions, descriptors = [], [], []
        try:
            for path, suffix in ((self.path, "weight"), (scale_path, "scale")):
                with path.open("rb") as stream:
                    raw = stream.read(8)
                    if len(raw) != 8:
                        raise ValueError("truncated safetensors prefix")
                    length, = struct.unpack("<Q", raw)
                    if not 2 <= length <= 128 * 1024 * 1024:
                        raise ValueError("safetensors header exceeds bound")
                    item = json.loads(stream.read(length))[prefix + "." + suffix]
                    descriptors.append(item)
                    start, end = item["data_offsets"]
                    rows, width = item["shape"]
                    if (rows <= 0 or width <= 0 or start < 0 or end - start != rows * width
                            or 8 + length + end > path.stat().st_size):
                        raise ValueError("invalid PLE payload extent")
                    regions.append((8 + length + start, width))
                    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    mappings.append(mapping)
                    mapping.madvise(mmap.MADV_RANDOM)
            weight, scale = descriptors
            self.rows, self.width = weight["shape"]
            if weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F8_E8M0":
                raise ValueError("PLE requires FP8 E4M3 values and E8M0 scales")
            if self.width % 32 or scale["shape"] != [self.rows, self.width // 32]:
                raise ValueError("PLE scale shape mismatch")
            self._mapping, self._scale_table = mappings
            self._regions = regions
        except BaseException:
            for mapping in mappings:
                mapping.close()
            self._mapping = self._scale_table = None
            raise

    def _mapped_regions(self):
        return zip((self._mapping, self._scale_table if self._scale_table is not None else self._mapping),
                   self._regions)

    def _indices(self, indices):
        result = []
        for index in indices:
            if len(result) >= self.max_gather_rows:
                raise ValueError("PLE gather exceeds bounded row count")
            index = operator.index(index)
            if not 0 <= index < self.rows:
                raise IndexError("PLE row outside table")
            result.append(index)
        if self._mapping is None:
            raise RuntimeError("PLE table is closed")
        return result

    def prefetch(self, indices):
        """Advise only pages needed by a bounded upcoming row batch."""
        indices = self._indices(indices)
        page_size = mmap.PAGESIZE
        for mapping, (offset, width) in self._mapped_regions():
            pages = set()
            for index in indices:
                start = offset + index * width
                pages.update(range(start // page_size, (start + width - 1) // page_size + 1))
            for page in sorted(pages):
                start = page * page_size
                mapping.madvise(mmap.MADV_WILLNEED, start,
                                min(page_size, len(mapping) - start))

    def gather(self, indices):
        """Return owned (FP8 bytes, E8M0 bytes), preserving row order/repeats."""
        indices = self._indices(indices)
        return tuple(b"".join(mapping[offset + index * width:
                                           offset + (index + 1) * width]
                              for index in indices)
                     for mapping, (offset, width) in self._mapped_regions())

    def release(self):
        if self._scale_table is not None:
            self._scale_table.madvise(mmap.MADV_DONTNEED)
        if self._mapping is not None:
            self._mapping.madvise(mmap.MADV_DONTNEED)

    def close(self):
        if self._scale_table is not None:
            self._scale_table.close()
            self._scale_table = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
