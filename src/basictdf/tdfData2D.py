__doc__ = """
Data2D module
"""
from enum import Enum

import numpy as np
from datetime import datetime

from basictdf.tdfBlock import Block, BlockType
from basictdf.tdfTypes import (
    VEC2F,
    BTSDate,
    i32,
    i16,
    u32,
    u16,
)


class Data2DFlags(Enum):
    with_distortion = 0
    without_distortion = 1


class Data2DPCK:
    def _build(stream, nFrames, nCameras):
        nPointsCaptured = u16.bread(stream, nCameras * nFrames).reshape(
            [nCameras, nFrames]
        )
        data = np.empty((nFrames, nCameras), dtype=object)

        for frame in range(nFrames):
            for camera in range(nCameras):
                nPoints = nPointsCaptured[camera, frame]
                if nPoints > 0:
                    data[frame, camera] = VEC2F.bread(stream, nPoints)
        return data

    def _write(stream, data) -> None:
        nFrames, nCameras = data.shape
        nPointsCaptured = np.zeros((nCameras, nFrames), dtype=np.uint16)
        for frame in range(nFrames):
            for camera in range(nCameras):
                if data[frame, camera] is not None:
                    nPointsCaptured[camera, frame] = len(data[frame, camera])
        u16.bwrite(stream, nPointsCaptured.flatten())
        for frame in range(nFrames):
            for camera in range(nCameras):
                if data[frame, camera] is not None:
                    VEC2F.bwrite(stream, data[frame, camera])


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
        startTime: datetime,
        flags: Data2DFlags,
        data: np.ndarray,
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
        self.data = data
        """a numpy array of shape (nFrames, nCams).
        Each element is a numpy array of shape (nPoints, 2) containing the 2D coordinates of the points
        captured by the camera at the corresponding frame.
        """
        self.format = format

        self._camMap = []

    @staticmethod
    def _build(stream, format) -> "Data2D":
        format = Data2DBlockFormat(format)
        if format != Data2DBlockFormat.PCKFormat:
            raise NotImplementedError(f"Data2D format {format} is not implemented yet.")
        nCams = i32.bread(stream)
        nFrames = i32.bread(stream)
        frequency = i32.bread(stream)
        startTime = BTSDate.bread(stream)
        flags = Data2DFlags(u32.bread(stream))
        camMap = u16.bread(stream, nCams)
        data = Data2DPCK._build(stream, nFrames, nCams)

        block = Data2D(nCams, nFrames, frequency, startTime, flags, data, format)
        block._camMap = camMap

        return block

    def _write(self, stream) -> None:
        if self.format != Data2DBlockFormat.PCKFormat:
            raise NotImplementedError(
                f"Writing Data2D format {self.format} is not implemented yet."
            )
        i32.bwrite(stream, self.nCams)
        i32.bwrite(stream, self.nFrames)
        i32.bwrite(stream, self.frequency)
        BTSDate.bwrite(stream, self.startTime)
        u32.bwrite(stream, self.flags.value)
        i16.bwrite(stream, self._camMap)
        Data2DPCK._write(stream, self.data)

    @property
    def nBytes(self):
        return (
            i32.btype.itemsize
            + i32.btype.itemsize
            + i32.btype.itemsize
            + 4  # BTSDate.nBytes
            + u32.btype.itemsize
            + i16.btype.itemsize * self.nCams
            + self.data.nbytes
        )
