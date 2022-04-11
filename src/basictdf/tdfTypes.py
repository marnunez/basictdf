__doc__ = "Shared classes and types for basictdf."

from datetime import datetime
import struct
import numpy as np


class BTSDate:
    @staticmethod
    def read(data):
        return datetime.fromtimestamp(struct.unpack("<i", data)[0])

    @staticmethod
    def bread(f):
        return BTSDate.read(f.read(4))

    @staticmethod
    def write(data):
        return struct.pack("<i", int(data.timestamp()))

    @staticmethod
    def bwrite(file, data):
        file.write(BTSDate.write(data))


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

    @staticmethod
    def bwrite(file, size, data):
        file.write(BTSString.write(size, data))

    @staticmethod
    def bread(f, size):
        return BTSString.read(size, f.read(size))


class Type:
    def __init__(self, btype):
        self.btype = btype

    def read(self, data, n=-1):
        if n == -1:
            return np.frombuffer(data, dtype=self.btype)[0]
        else:
            return np.frombuffer(data, dtype=self.btype)

    def bread(self, file, n=-1):
        return self.read(file.read(abs(n) * self.btype.itemsize), n=n)

    def write(self, data):
        return (
            data.astype(self.btype.base).tobytes()
            if isinstance(data, np.ndarray)
            else np.array(data, dtype=self.btype.base).tobytes()
        )

    def bwrite(self, file, data):
        file.write(self.write(data))

    def skip(self, file, n=1):
        file.seek(n * self.btype.itemsize, 1)

    def pad(self, n=1):
        return b"\x00" * (n * self.btype.itemsize)

    def bpad(self, file, n=1):
        file.write(self.pad(n))


Volume = Type(np.dtype("3<f4"))

VEC3F = Type(np.dtype("3<f4"))

Matrix = Type(np.dtype("(3,3)<f4"))


Int32 = Type(np.dtype("<i4"))
Int16 = Type(np.dtype("<i2"))
Uint32 = Type(np.dtype("<u4"))
Float32 = Type(np.dtype("<f4"))
