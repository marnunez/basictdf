import struct
from datetime import datetime
import numpy as np


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class BTSDate:
    @staticmethod
    def read(data):
        return datetime.fromtimestamp(struct.unpack("<I", data)[0])

    @staticmethod
    def write(data):
        return struct.pack("<I", data.timestamp())


class BTSString:
    """
    BTSString is a null terminated string with a length prefix.
    """

    @staticmethod
    def read(size, data):
        la = struct.unpack(f"{size}s", data)[0]
        try:
            pos = la.index(b"\x00")
            return la[:pos].decode("windows-1252")
        except ValueError:
            return la.decode("windows-1252")

    @staticmethod
    def write(size, data):
        dat = data.encode("windows-1252") + b"\x00"
        padding = b"\x00" * (size - len(dat))
        if len(dat) > size:
            raise ValueError(f"data is too long for {size}")
        return dat + padding


class Type:
    def __init__(self, btype):
        self.btype = btype

    def read(self, file, n):
        return np.frombuffer(file.read(n * self.btype.itemsize), dtype=self.btype)

    def write(self, data):
        return data.tobytes()

    def skip(self, file, n):
        file.seek(n * self.btype.itemsize, 1)

    def pad(self, n):
        return b"\x00" * (n * self.btype.itemsize)


Volume = Type(np.dtype([("length", "<f4"), ("width", "<f4"), ("depth", "<f4")]))


# nVector = Type(np.dtype("3<f4"))
VEC3F = Type(np.dtype([("X", "<f4"), ("Y", "<f4"), ("Z", "<f4")]))

Matrix = Type(np.dtype("(3,3)<f4"))


Int32 = Type(np.dtype("<i4"))
Int16 = Type(np.dtype("<i2"))
Uint32 = Type(np.dtype("<u4"))
Float32 = Type(np.dtype("<f4"))
