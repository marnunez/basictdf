__doc__ = "Shared classes and types for basictdf."

from datetime import datetime
import struct
from typing import (
    IO,
    Any,
    Optional,
    Type,
    TypeVar,
    Union,
    BinaryIO,
    Generic,
)
import numpy as np
import numpy.typing as npt


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
            raise ValueError(f"data is too long for {size}")
        return dat + padding

    @staticmethod
    def bwrite(file: BinaryIO, size: int, data: str):
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
    def __init__(self, btype: npt.DTypeLike):
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
            n (int, optional): Ammount of items to take. If _None_, returns a single item, otherwise returns an array of _n_ items. Defaults to None.

        Returns:
            Union[np.ndarray,type]: A numpy type (custom or classic, like numpy.float32) or a np.array of numpy types
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

    def bwrite(self, file: IO[bytes], data: Union[npt.NDArray[X], X]):
        file.write(self.write(data))

    def skip(self, file: IO[bytes], n: int = 1):
        file.seek(n * self.btype.itemsize, 1)

    def pad(self, n: int = 1):
        return b"\x00" * (n * self.btype.itemsize)

    def bpad(self, file: IO[bytes], n: int = 1):
        file.write(self.pad(n))


Volume = TdfType(np.dtype("3<f4"))

VEC3F = TdfType(np.dtype("3<f4"))

Matrix = TdfType(np.dtype("(3,3)<f4"))


Int32 = TdfType(np.dtype("<i4"))
Int16 = TdfType(np.dtype("<i2"))
Uint32 = TdfType(np.dtype("<u4"))
Float32 = TdfType(np.dtype("<f4"))
