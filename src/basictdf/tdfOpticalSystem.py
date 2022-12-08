__doc__ = """
Optical System Configuration Data module.
"""
from enum import Enum
from io import BytesIO
from typing import BinaryIO, Iterator, List, Union
from basictdf.tdfBlock import Block, BlockType
from basictdf.tdfTypes import BTSString, Int32, Int16, VEC2I, TdfType
import numpy as np


class CameraViewPort:
    def __init__(self, origin, size) -> None:
        if isinstance(origin, np.ndarray) and origin.shape != VEC2I.btype.shape:
            raise TypeError(
                f"origin must be a {VEC2I.btype.shape} if it is a numpy array"
            )
        elif isinstance(origin, list) or isinstance(origin, tuple) and len(origin) != 2:
            raise TypeError(f"origin must be of length == 2 if it is a list or tuple")

        if isinstance(size, np.ndarray) and size.shape != VEC2I.btype.shape:
            raise TypeError(
                f"size must be a {VEC2I.btype.shape} if it is a numpy array"
            )
        elif isinstance(size, list) or isinstance(size, tuple) and len(size) != 2:
            raise TypeError(f"size must be of length == 2 if it is a list or tuple")

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

    @property
    def nBytes(self) -> int:
        return 8 + 8

    def __repr__(self) -> str:
        return f"CameraViewPort(origin={self.origin}, size={self.size})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CameraViewPort):
            raise TypeError(f"Can only compare CameraViewPort with CameraViewPort")
        return np.all(self.origin == other.origin) and np.all(self.size == other.size)


class OpticalChannelData:
    """
    An object that collects data for a single channel of the optical system.
    """

    def __init__(
        self,
        logical_camera_index: int,
        lens_name: str,
        camera_type: str,
        camera_name: str,
        camera_viewport: Union[CameraViewPort, np.ndarray],
    ) -> None:
        self.logical_camera_index = logical_camera_index
        "Logical index of the camera. Used to define sorting"
        self.lens_name = lens_name
        "Lens name"
        self.camera_type = camera_type
        "Camera type"
        self.camera_name = camera_name
        "Camera name / symbolic camera number"

        if isinstance(camera_viewport, CameraViewPort):
            camera_viewport = camera_viewport
        elif isinstance(camera_viewport, np.ndarray) and camera_viewport.shape == (
            2,
            2,
        ):
            camera_viewport = CameraViewPort(camera_viewport[0], camera_viewport[1])
        else:
            raise TypeError(
                f"camera_viewport must be a CameraViewPort or a (2,2) shape numpy array"
            )

        self.camera_viewport = camera_viewport
        "Camera viewport"

    @staticmethod
    def _build(stream) -> "OpticalChannelData":
        logical_index = Int32.bread(stream)
        Int32.skip(stream)  # reserved0
        lens_name = BTSString.bread(stream, 32)
        camera_type = BTSString.bread(stream, 32)
        camera_name = BTSString.bread(stream, 32)
        camera_viewport = CameraViewPort.bread(stream)
        return OpticalChannelData(
            logical_camera_index=logical_index,
            lens_name=lens_name,
            camera_type=camera_type,
            camera_name=camera_name,
            camera_viewport=camera_viewport,
        )

    def _write(self, file) -> None:

        # logical camera index
        Int32.bwrite(file, self.logical_camera_index)

        # Reserved 0
        Int32.bpad(file, 1)

        # lens name
        BTSString.bwrite(file, 32, self.lens_name)

        # camera type
        BTSString.bwrite(file, 32, self.camera_type)

        # camera name
        BTSString.bwrite(file, 32, self.camera_name)

        # camera viewport
        self.camera_viewport.bwrite(file)

    @property
    def nBytes(self) -> int:
        """
        Returns:
            int: size of the track in bytes
        """

        return 4 + 4 + 32 + 32 + 32 + self.camera_viewport.nBytes

    def __repr__(self) -> str:
        return f"OpticalChannelData(camera_name={self.camera_name}, camera_type={self.camera_type})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, OpticalChannelData):
            raise TypeError(
                "Can only compare OpticalChannelData with OpticalChannelData"
            )
        return (
            self.logical_camera_index == other.logical_camera_index
            and self.lens_name == other.lens_name
            and self.camera_type == other.camera_type
            and self.camera_name == other.camera_name
            and self.camera_viewport == other.camera_viewport
        )


class OpticalSetupBlockFormat(Enum):
    unknownFormat = 0
    basicFormat = 1


class OpticalSetupBlock(Block):
    type = BlockType.opticalSystemConfiguration

    def __init__(
        self,
        format: OpticalSetupBlockFormat = OpticalSetupBlockFormat.basicFormat,
        channels: List[OpticalChannelData] = [],
    ) -> None:
        """A data block containing information about the physical setup of
        motion capture.
        """

        super().__init__()
        self.format = format
        self.channels = channels

    @staticmethod
    def _build(stream, format) -> "OpticalSetupBlock":
        format = OpticalSetupBlockFormat(format)
        nChannels = Int32.bread(stream)
        Int32.skip(stream)  # reserved0

        channels = [OpticalChannelData._build(stream) for _ in range(nChannels)]
        return OpticalSetupBlock(format=format, channels=channels)

    def __iter__(self) -> Iterator[OpticalChannelData]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __eq__(self, other: "OpticalSetupBlock") -> bool:
        buff1 = BytesIO()
        buff2 = BytesIO()
        self._write(buff1)
        other._write(buff2)
        return buff1.getvalue() == buff2.getvalue()

    def __contains__(self, value: OpticalChannelData) -> bool:
        if isinstance(value, OpticalChannelData):
            return value in self.channels
        raise TypeError(f"Invalid value type {type(value)}")

    @property
    def nChannels(self) -> int:
        """Number of channels in the optical setup block

        Returns:
            int: number of channels
        """
        return len(self.channels)

    def _write(self, file: BinaryIO) -> None:

        # nChannels
        Int32.bwrite(file, len(self.channels))

        # Reserved 0
        Int32.bpad(file, 1)

        # channels
        for channel in self.channels:
            channel._write(file)

    @property
    def nBytes(self) -> int:
        base = 4 + 4

        for channel in self.channels:
            base += channel.nBytes

        return base

    def __repr__(self) -> str:
        return f"<OpticalSetupBlock: {len(self.channels)} channels>"
