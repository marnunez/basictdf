__doc__ = "Force, torque and acceleration data module."

from io import BytesIO
from re import I
from basictdf.tdfBlock import Block, BlockType
from enum import Enum
from basictdf.tdfData3D import TrackType
from basictdf.tdfTypes import (
    Int32,
    Float32,
    BTSString,
    BTSDate,
    Uint32,
    VEC3F,
    Matrix,
    Volume,
    Type,
)
import numpy as np

SegmentData = Type(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))
ForceType = ApplicationPointType = TorqueType = Type(np.dtype("<3f4"))


class ForceTorqueTrack:
    def __init__(self, label, applicationPoint, forceData, torqueData):
        self.label = label
        self.applicationPoint = applicationPoint
        self.force = forceData
        self.torque = torqueData

    @property
    def _segments(self):
        maskedPressureData = np.ma.masked_invalid(self.applicationPoint)
        return np.ma.clump_unmasked(maskedPressureData.T[0])

    @property
    def nFrames(self):
        return self.applicationPoint.shape[0]

    @property
    def nBytes(self):
        base = 256 + 4 + 4
        for segment in self._segments:
            base += (
                4
                + 4
                + (segment.stop - segment.start)
                * (
                    ApplicationPointType.btype.itemsize
                    + ForceType.btype.itemsize
                    + TorqueType.btype.itemsize
                )
            )
        return base

    @staticmethod
    def build(stream, nFrames):
        # label
        label = BTSString.bread(stream, 256)
        print(label)
        # nSegments
        nSegments = Int32.bread(stream)
        # padding
        Int32.skip(stream)

        segmentData = SegmentData.bread(stream, nSegments)

        applicationPointData = np.empty(nFrames, dtype=ApplicationPointType.btype)
        applicationPointData[:] = np.nan

        forceData = np.empty(nFrames, dtype=ForceType.btype)
        forceData[:] = np.nan

        torqueData = np.empty(nFrames, dtype=TorqueType.btype)
        torqueData[:] = np.nan

        for startFrame, nFrames in segmentData:
            applicationPointData[
                startFrame : startFrame + nFrames
            ] = ApplicationPointType.bread(stream, nFrames)
            forceData[startFrame : startFrame + nFrames] = ForceType.bread(
                stream, nFrames
            )
            torqueData[startFrame : startFrame + nFrames] = TorqueType.bread(
                stream, nFrames
            )
        return ForceTorqueTrack(label, forceData, applicationPointData, torqueData)

    def write(self, file):

        # label
        BTSString.bwrite(file, 256, self.label)

        segments = self._segments

        # nSegments
        Int32.bwrite(file, len(segments))

        # padding
        Int32.bpad(file)

        # segmentData
        SegmentData.bwrite(file, segments)

        for segment in segments:
            # startFrame
            Int32.bwrite(file, segment.start)
            # nFrames
            Int32.bwrite(file, segment.stop - segment.start)
            # applicationPoint
            ApplicationPointType.bwrite(file, self.applicationPoint[segment])
            # force
            ForceType.bwrite(file, self.force[segment])
            # torque
            TorqueType.bwrite(file, self.torque[segment])

    def __repr__(self):
        return f"ForceTorqueTrack(label={self.label}, nFrames={self.nFrames})"

    def __eq__(self, other):
        return (
            self.label == other.label
            and np.all(self.applicationPoint == other.applicationPoint)
            and np.all(self.force == other.force)
            and np.all(self.torque == other.torque)
        )


class Force3DBlockFormat(Enum):
    unknownFormat = 0
    byTrack = 1
    byFrame = 2
    byTrackWithSpeed = 3
    byFrameWithSpeed = 4


class ForceTorque3D(Block):
    def __init__(
        self,
        frequency,
        nFrames,
        volume,
        rotationMatrix,
        translationVector,
        startTime=0.0,
        format=Force3DBlockFormat.byTrack,
    ):
        super().__init__(BlockType.forceAndTorqueData)
        self.format = format
        self.frequency = frequency
        self.startTime = startTime

        if not (
            isinstance(rotationMatrix, np.ndarray)
            and rotationMatrix.shape == Matrix.btype.shape
        ):
            raise ValueError(
                f"rotationMatrix must be a numpy array of shape {Matrix.btype.shape}"
            )
        self.rotationMatrix = rotationMatrix

        if not (
            isinstance(translationVector, np.ndarray)
            and translationVector.shape == VEC3F.btype.shape
        ):
            raise ValueError(
                f"translationVector must be a numpy array of shape {VEC3F.btype}"
            )
        self.translationVector = translationVector

        if not (isinstance(volume, np.ndarray) and volume.shape == Volume.btype.shape):
            raise ValueError(
                f"volume must be a numpy array of shape {Volume.btype.shape}"
            )
        self.volume = volume

        self.nFrames = nFrames
        self._tracks = []

    @staticmethod
    def build(stream, format):
        format = Force3DBlockFormat(format)
        nTracks = Int32.bread(stream)
        frequency = Int32.bread(stream)
        startTime = Float32.bread(stream)
        nFrames = Uint32.bread(stream)
        volume = Volume.bread(stream)
        rotationMatrix = Matrix.bread(stream)
        translationVector = VEC3F.bread(stream)

        Int32.skip(stream)
        f = ForceTorque3D(
            frequency,
            nFrames,
            volume,
            rotationMatrix,
            translationVector,
            startTime,
            format,
        )
        if format != Force3DBlockFormat.byTrack:
            raise NotImplementedError(f"Force3D format {format} not implemented yet")

        f._tracks = [ForceTorqueTrack.build(stream, nFrames) for _ in range(nTracks)]
        return f

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._tracks[key]
        elif isinstance(key, str):
            try:
                return next(track for track in self._tracks if track.label == key)
            except StopIteration:
                raise KeyError(f"Track with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __contains__(self, key):
        if isinstance(key, str):
            return any(track.label == key for track in self._tracks)
        elif isinstance(key, ForceTorqueTrack):
            return key in self._tracks
        raise TypeError(f"Invalid key type {type(key)}")

    def __iter__(self):
        return iter(self._tracks)

    def __len__(self):
        return len(self._tracks)

    def __eq__(self, other):
        buff1 = BytesIO()
        buff2 = BytesIO()
        self.write(buff1)
        other.write(buff2)
        return buff1.getvalue() == buff2.getvalue()

    @property
    def nTracks(self):
        return len(self._tracks)

    def write(self, file):
        if self.format != Force3DBlockFormat.byTrack:
            raise NotImplementedError(
                f"Force3D format {self.format} not implemented yet"
            )
        # nFrames
        Int32.bwrite(file, self.nFrames)

        # frequency
        Int32.bwrite(file, self.frequency)
        # startTime
        Float32.bwrite(file, self.startTime)
        # nTracks
        Uint32.bwrite(file, len(self._tracks))

        # volume
        Volume.bwrite(file, self.volume)
        # rotationMatrix
        Matrix.bwrite(file, self.rotationMatrix)
        # translationVector
        VEC3F.bwrite(file, self.translationVector)

        # padding
        Int32.bpad(file)

        for track in self._tracks:
            track.write(file)

    def __repr__(self):
        return f"<ForceTorque3D: {self.nFrames} frames, {self.frequency} Hz, {self.nTracks} tracks, tracks={[i.label for i in self._tracks]}>"
