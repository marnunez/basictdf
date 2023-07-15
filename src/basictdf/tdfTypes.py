__doc__ = "Shared classes and types for basictdf."

import struct
from datetime import datetime
from typing import IO, BinaryIO, Generic, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt


class BTSDate:
    @staticmethod
    def read(data) -> datetime:
        return datetime.fromtimestamp(struct.unpack("<i", data)[0])

    @staticmethod
    def bread(f) -> datetime:
        return BTSDate.read(f.read(4))

    @staticmethod
    def write(data) -> bytes:
        return struct.pack("<i", int(data.timestamp()))

    @staticmethod
    def bwrite(file, data) -> None:
        file.write(BTSDate.write(data))


class BTSString:
    """
    BTSString is a null terminated string with a length prefix.
    """

    @staticmethod
    def read(size: int, data: bytes, encoding: str = "windows-1252") -> str:
        """read a BTSString from bytes

        Args:
            size (int): size of the string to read
            data (bytes): input bytes
            encoding (str, optional): encoding to use. Defaults to "windows-1252".

        Returns:
            str: a Python string
        """
        la = struct.unpack(f"{size}s", data)[0]
        try:
            pos = la.index(b"\x00")
            return la[:pos].decode(encoding)
        except ValueError:
            return la.decode(encoding)

    @staticmethod
    def write(size: int, data: str) -> bytes:
        dat = data.encode("windows-1252") + b"\x00"
        padding = b"\x00" * (size - len(dat))
        if len(dat) > size:
            raise ValueError(
                f"The string is too long: max {size} chars, got {len(dat)}"
            )
        return dat + padding

    @staticmethod
    def bwrite(file: BinaryIO, size: int, data: str) -> None:
        file.write(BTSString.write(size, data))

    @staticmethod
    def bread(file: BinaryIO, size: int, encoding: str = "windows-1252") -> str:
        """Read a BTSString from a binary file or buffer

        Args:
            file (BinaryIO): input binary file or buffer
            size (int): size of the string to read
            encoding (str, optional): encoding to use. Defaults to "windows-1252".

        Returns:
            str: a Python string
        """
        return BTSString.read(size, file.read(size), encoding=encoding)


# T = NewType("T", np.dtype)
# T = TypeAlias(npt.DTypeLike)
# T = TypeVar("T", bound=npt.DTypeLike)
X = TypeVar("X", bound=np.dtype)


class TdfType(Generic[X]):
    def __init__(self, btype: npt.DTypeLike) -> None:
        self.btype: X = np.dtype(btype)

    def read(self, data: bytes) -> npt.NDArray[X]:
        """Read data to the type

        Args:
            data (bytes): input bytes

        Returns:
            np.ndarray: output array with items of the requiered type
        """
        return np.frombuffer(data, dtype=self.btype)

    def bread(
        self, file: IO[bytes], n: Optional[int] = None
    ) -> Union[npt.NDArray[X], X]:
        """Read data from binary file or buffer

        Args:
            file (IO[Any]): input file or buffer
            n (int, optional): Ammount of items to take. If _None_, returns a
            single item, otherwise returns an array of _n_ items. Defaults to None.

        Returns:
            Union[np.ndarray,type]: A numpy type (custom or classic,
            like numpy.float32) or a np.array of numpy types
        """
        if n is None:
            return self.read(file.read(self.btype.itemsize))[0]
        else:
            return self.read(file.read(n * self.btype.itemsize))

    def write(self, data: Union[npt.NDArray[X], X]):
        return (
            data.astype(self.btype.base).tobytes()
            if isinstance(data, np.ndarray)
            else np.array(data, dtype=self.btype.base).tobytes()
        )

    def bwrite(self, file: IO[bytes], data: Union[npt.NDArray[X], X]) -> None:
        file.write(self.write(data))

    def skip(self, file: IO[bytes], n: int = 1) -> None:
        file.seek(n * self.btype.itemsize, 1)

    def pad(self, n: int = 1):
        return b"\x00" * (n * self.btype.itemsize)

    def bpad(self, file: IO[bytes], n: int = 1) -> None:
        file.write(self.pad(n))

    def nBytes(self, n: int = 1) -> int:
        return n * self.btype.itemsize


Volume = TdfType(np.dtype("3<f4"))

VEC3F = TdfType(np.dtype("3<f4"))
VEC3D = TdfType(np.dtype("3<f8"))

VEC2I = TdfType(np.dtype("2<i4"))
VEC2F = TdfType(np.dtype("2<f4"))
VEC2D = TdfType(np.dtype("2<f8"))

MAT3X3F = TdfType(np.dtype("(3,3)<f4"))
MAT3X3D = TdfType(np.dtype("(3,3)<f8"))

i32 = TdfType(np.dtype("<i4"))
i16 = TdfType(np.dtype("<i2"))
u32 = TdfType(np.dtype("<u4"))
u16 = TdfType(np.dtype("<u2"))
f32 = TdfType(np.dtype("<f4"))
f64 = TdfType(np.dtype("<f8"))


class CameraViewPort:
    def __init__(self, origin, size) -> None:
        if isinstance(origin, np.ndarray) and origin.shape != VEC2I.btype.shape:
            raise TypeError(
                f"origin must be a {VEC2I.btype.shape} if it is a numpy array"
            )
        elif isinstance(origin, list) or isinstance(origin, tuple) and len(origin) != 2:
            raise TypeError("origin must be of length 2 if it is a list or tuple")

        if isinstance(size, np.ndarray) and size.shape != VEC2I.btype.shape:
            raise TypeError(
                f"size must be a {VEC2I.btype.shape} if it is a numpy array"
            )
        elif isinstance(size, list) or isinstance(size, tuple) and len(size) != 2:
            raise TypeError("size must be of length 2 if it is a list or tuple")

        self.origin = origin
        self.size = size

    @staticmethod
    def bread(stream) -> "CameraViewPort":
        origin = VEC2I.bread(stream)
        size = VEC2I.bread(stream)
        return CameraViewPort(origin, size)

    @staticmethod
    def read(data: bytes) -> "CameraViewPort":
        origin = VEC2I.read(data[:8])
        size = VEC2I.read(data[8:])
        return CameraViewPort(origin, size)

    def write(self) -> bytes:
        return VEC2I.write(self.origin) + VEC2I.write(self.size)

    def bwrite(self, stream: BinaryIO) -> None:
        VEC2I.bwrite(stream, self.origin)
        VEC2I.bwrite(stream, self.size)

    nBytes = 8 + 8

    def __repr__(self) -> str:
        return f"CameraViewPort(origin={self.origin}, size={self.size})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CameraViewPort):
            raise TypeError("Can only compare CameraViewPort with CameraViewPort")
        return np.array_equal(self.origin, other.origin) and np.array_equal(
            self.size, other.size
        )


SegmentData = TdfType(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))
