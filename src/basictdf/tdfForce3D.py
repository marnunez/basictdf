__doc__ = "Force, torque and acceleration data module."

from io import BytesIO
from re import I
from typing import BinaryIO, Iterable, Iterator, List, Optional, Type, Union
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
    TdfType,
)
import numpy as np

SegmentData = TdfType(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))
ForceType = ApplicationPointType = TorqueType = TdfType(np.dtype("<3f4"))


class ForceTorqueTrack:
    def __init__(
        self,
        label: str,
        application_point: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray,
    ):
        if (
            application_point.shape != force.shape
            or application_point.shape != torque.shape
        ):
            raise ValueError(
                "application_point, force and torque must have the same shape"
            )
        self.label = label
        self.application_point = application_point
        self.force = force
        self.torque = torque

    @property
    def _segments(self):
        maskedPressureData = np.ma.masked_invalid(self.application_point)
        return np.ma.clump_unmasked(maskedPressureData.T[0])

    @_segments.setter
    def _segments(self, value):
        raise AttributeError(
            "Can't set number of segments directly, it's inferred from the data"
        )

    @property
    def nFrames(self) -> int:
        return self.application_point.shape[0]

    @property
    def nBytes(self) -> int:
        segments = self._segments
        base = 256 + 4 + 4 + SegmentData.btype.itemsize * len(segments)
        for segment in segments:
            base += (segment.stop - segment.start) * (
                ApplicationPointType.btype.itemsize
                + ForceType.btype.itemsize
                + TorqueType.btype.itemsize
            )
        return base

    @staticmethod
    def _build(stream, frames: int) -> "ForceTorqueTrack":

        # label
        label = BTSString.bread(stream, 256)
        # nSegments
        nSegments = Int32.bread(stream)
        # padding
        Int32.skip(stream)

        segmentData = SegmentData.bread(stream, nSegments)

        application_point_data = np.empty(frames, dtype=ApplicationPointType.btype)
        application_point_data[:] = np.nan

        force_data = np.empty(frames, dtype=ForceType.btype)
        force_data[:] = np.nan

        torque_data = np.empty(frames, dtype=TorqueType.btype)
        torque_data[:] = np.nan

        for startFrame, nFrames in segmentData:
            application_point_data[
                startFrame : startFrame + nFrames
            ] = ApplicationPointType.bread(stream, nFrames)
            force_data[startFrame : startFrame + nFrames] = ForceType.bread(
                stream, nFrames
            )
            torque_data[startFrame : startFrame + nFrames] = TorqueType.bread(
                stream, nFrames
            )
        return ForceTorqueTrack(
            label=label,
            application_point=application_point_data,
            force=force_data,
            torque=torque_data,
        )

    def _write(self, file: BinaryIO):

        # label
        BTSString.bwrite(file, 256, self.label)

        segments = self._segments
        nSegments = len(segments)

        # nSegments
        Int32.bwrite(file, nSegments)

        # padding
        Int32.bpad(file)

        # segmentData
        for segment in segments:
            # startFrame
            Int32.bwrite(file, segment.start)
            # nFrames
            Int32.bwrite(file, segment.stop - segment.start)

        for segment in segments:
            # applicationPoint
            ApplicationPointType.bwrite(file, self.application_point[segment])
            # force
            ForceType.bwrite(file, self.force[segment])
            # torque
            TorqueType.bwrite(file, self.torque[segment])

    def __repr__(self) -> str:
        return f"ForceTorqueTrack(label={self.label}, nFrames={self.nFrames})"

    def __eq__(self, other: Type["ForceTorqueTrack"]) -> bool:
        return (
            self.label == other.label
            and np.all(self.application_point == other.application_point)
            and np.all(self.force == other.force)
            and np.all(self.torque == other.torque)
        )


class ForceTorque3DBlockFormat(Enum):
    unknownFormat = 0
    byTrack = 1
    byFrame = 2
    byTrackWithSpeed = 3
    byFrameWithSpeed = 4


class ForceTorque3D(Block):
    def __init__(
        self,
        frequency: Union[int, float],
        nFrames: int,
        volume: np.ndarray,
        rotationMatrix: np.ndarray,
        translationVector: np.ndarray,
        startTime: float = 0.0,
        format=ForceTorque3DBlockFormat.byTrack,
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
    def _build(stream, format):
        format = ForceTorque3DBlockFormat(format)
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
        if format != ForceTorque3DBlockFormat.byTrack:
            raise NotImplementedError(f"Force3D format {format} not implemented yet")

        f._tracks = [ForceTorqueTrack._build(stream, nFrames) for _ in range(nTracks)]
        return f

    def add_track(self, track: ForceTorqueTrack):
        """Adds a track to the data block

        Args:
            track (MarkerTrack): track to add

        Raises:
            TypeError: Track is not of type MarkerTrack
            ValueError: Track has a different number of frames than the data block
        """
        if not isinstance(track, ForceTorqueTrack):
            raise TypeError(f"Track must be of type ForceTorqueTrack")
        if track.nFrames != self.nFrames:
            raise ValueError(
                f"Track with label {track.label} has {track.nFrames} frames, expected {self.nFrames} frames"
            )
        self._tracks.append(track)

    @property
    def tracks(self) -> List[ForceTorqueTrack]:
        """Returns a list of all tracks in the data block

        Returns:
            List[MarkerTrack]: list of all tracks in the data block
        """
        return self._tracks

    @tracks.setter
    def tracks(self, values: Iterable[ForceTorqueTrack]):
        """
        Sets the tracks in the data block.
        """
        oldTracks = self._tracks
        self._tracks = []
        try:
            for value in values:
                self.add_track(value)
        except Exception as e:
            self._tracks = oldTracks
            raise e

    def __getitem__(self, key: Union[int, str]) -> ForceTorqueTrack:
        if isinstance(key, int):
            return self._tracks[key]
        elif isinstance(key, str):
            try:
                return next(track for track in self._tracks if track.label == key)
            except StopIteration:
                raise KeyError(f"Track with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __contains__(self, key: Union[str, ForceTorqueTrack]) -> bool:
        if isinstance(key, str):
            return any(track.label == key for track in self._tracks)
        elif isinstance(key, ForceTorqueTrack):
            return key in self._tracks
        raise TypeError(f"Invalid key type {type(key)}")

    def __iter__(self) -> Iterator[ForceTorqueTrack]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __eq__(self, other) -> bool:
        buff1 = BytesIO()
        buff2 = BytesIO()
        self._write(buff1)
        other._write(buff2)
        return buff1.getvalue() == buff2.getvalue()

    @property
    def nTracks(self) -> int:
        return len(self._tracks)

    @property
    def nBytes(self):
        base = (
            4
            + 4
            + 4
            + 4
            + Volume.btype.itemsize
            + Matrix.btype.itemsize
            + VEC3F.btype.itemsize
            + 4
        )
        for track in self._tracks:
            base += track.nBytes
        return base

    def _write(self, file):
        if self.format != ForceTorque3DBlockFormat.byTrack:
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
            track._write(file)

    def __repr__(self):
        return f"<ForceTorque3D: {self.nFrames} frames, {self.frequency} Hz, {self.nTracks} tracks, tracks={[i.label for i in self._tracks]}>"
