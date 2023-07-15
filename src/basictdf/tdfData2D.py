__doc__ = """
Data2D module
"""
from enum import Enum

import numpy as np

from basictdf.tdfBlock import Block, BlockType, Sized, BuildWriteable
from basictdf.tdfTypes import VEC2F, i32, i16, u32, u16, f32


class Data2DFlags(Enum):
    with_distortion = 0
    without_distortion = 1


class Data2DPCK(Sized, BuildWriteable):
    def __init__(self, data) -> None:
        self.data = data

    @staticmethod
    def _build(stream, nFrames, nCameras) -> "Data2DPCK":
        nPointsCaptured = u16.bread(stream, nCameras * nFrames).reshape(
            [nCameras, nFrames]
        )
        data = np.empty((nFrames, nCameras), dtype=object)

        for frame in range(nFrames):
            for camera in range(nCameras):
                nPoints = nPointsCaptured[camera, frame]
                if nPoints > 0:
                    data[frame, camera] = VEC2F.bread(stream, nPoints)
        return Data2DPCK(data)

    def _write(self, stream) -> None:
        nFrames, nCameras = self.data.shape
        nPointsCaptured = np.zeros((nCameras, nFrames), dtype=np.uint16)
        for frame in range(nFrames):
            for camera in range(nCameras):
                if self.data[frame, camera] is not None:
                    nPointsCaptured[camera, frame] = len(self.data[frame, camera])
        u16.bwrite(stream, nPointsCaptured.flatten())
        for frame in range(nFrames):
            for camera in range(nCameras):
                if self.data[frame, camera] is not None:
                    VEC2F.bwrite(stream, self.data[frame, camera])

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Data2DPCK):
            return False
        return np.allclose(self.data, o.data)

    @property
    def nBytes(self):
        nFrames, nCameras = self.data.shape
        return 2 * nCameras * nFrames + sum(
            self.data[i, j].nbytes
            for i in range(nFrames)
            for j in range(nCameras)
            if self.data[i, j] is not None
        )


class Data2DBlockFormat(Enum):
    unknownFormat = 0
    RTSFormat = 1  # TDF_DATA2D_FORMAT_RTS
    PCKFormat = 2  # TDF_DATA2D_FORMAT_PCK
    SYNCFormat = 3  # TDF_DATA2D_FORMAT_SYNC


class Data2D(Block):
    """
    Class that stores the 2D data of a capture.
    """

    type = BlockType.data2D

    def __init__(
        self,
        nCams: int,
        nFrames: int,
        frequency: int,
        startTime: float,
        flags: Data2DFlags,
        format: Data2DBlockFormat = Data2DBlockFormat.PCKFormat,
    ) -> None:
        super().__init__()
        self.nCams = nCams
        "number of cameras"
        self.nFrames = nFrames
        "number of frames captured"
        self.frequency = frequency
        "frequency of the capture in Hz"
        self.startTime = startTime
        "start time of the capture"
        self.flags: Data2DFlags = flags
        """a numpy array of shape (nFrames, nCams).
        Each element is a numpy array of shape (nPoints, 2) containing
        the 2D coordinates of the points captured by the camera at
        the corresponding frame.
        """
        self.format = format

        self._camMap = []
        self._data = None

    def __iter__(self):
        return iter(self._data.data)

    @property
    def data(self):
        return self._data.data

    @data.setter
    def data(self, value) -> None:
        self._data = Data2DPCK(value)

    @staticmethod
    def _build(stream, format) -> "Data2D":
        format = Data2DBlockFormat(format)
        if format != Data2DBlockFormat.PCKFormat:
            raise NotImplementedError(f"Data2D format {format} is not implemented yet.")
        nCams = i32.bread(stream)
        nFrames = i32.bread(stream)
        frequency = i32.bread(stream)
        startTime = f32.bread(stream)
        flags = Data2DFlags(u32.bread(stream))
        camMap = u16.bread(stream, nCams)
        data = Data2DPCK._build(stream, nFrames, nCams)

        block = Data2D(nCams, nFrames, frequency, startTime, flags, format)
        block._camMap = camMap
        block._data = data

        return block

    def _write(self, stream) -> None:
        if self.format != Data2DBlockFormat.PCKFormat:
            raise NotImplementedError(
                f"Writing Data2D format {self.format} is not implemented yet."
            )
        i32.bwrite(stream, self.nCams)
        i32.bwrite(stream, self.nFrames)
        i32.bwrite(stream, self.frequency)
        f32.bwrite(stream, self.startTime)
        u32.bwrite(stream, self.flags.value)
        i16.bwrite(stream, self._camMap)
        self._data._write(stream)

    @property
    def nBytes(self):
        return (
            4  # nCams
            + 4  # nFrames
            + 4  # frequency
            + 4  # BTSDate.nBytes
            + 4  # flags
            + 2 * self.nCams  # camMap
            # Size of the data without the None elements
            + self._data.nBytes
        )

    def __repr__(self) -> str:
        return (
            f"<Data2D format={self.format.name} nCams={self.nCams} "
            f"nFrames={self.nFrames} frequency={self.frequency} "
            f"startTime={self.startTime} flags={self.flags.name} >"
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Data2D):
            return False
        return (
            self.nCams == o.nCams
            and self.nFrames == o.nFrames
            and self.frequency == o.frequency
            and self.startTime == o.startTime
            and self.flags == o.flags
            and all(
                np.array_equal(self.data[i, j], o.data[i, j])
                for i in range(self.nFrames)
                for j in range(self.nCams)
            )
            and self.format == o.format
            and self.nBytes == o.nBytes
        )
