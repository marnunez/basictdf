__doc__ = """
Marker data module.
"""
from enum import Enum
from io import BytesIO
from typing import BinaryIO, Iterable, Iterator, List, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType, BuildWriteable, Sized
from basictdf.tdfTypes import (
    MAT3X3F,
    VEC3F,
    BTSString,
    TdfType,
    Volume,
    f32,
    i32,
    u32,
    SegmentData,
)


class Data3dBlockFormat(Enum):
    """
    Available formats for a Data3D block.
    """

    unknownFormat = 0
    byTrack = 1
    byTrackWithoutLinks = 2
    byFrame = 3
    byFrameWithoutLinks = 4


class Flags(Enum):
    """
    Available data flags for a Data3D block.
    """

    rawData = 0
    filtered = 1


LinkType = TdfType(np.dtype([("Track1", "<u4"), ("Track2", "<u4")]))

TrackType = TdfType(np.dtype("<3f4"))


class MarkerTrack(Sized, BuildWriteable):
    """
    A track that collects all the data of a physical marker, such as name and position.
    """

    def __init__(self, label: str, track_data: np.ndarray) -> None:
        self.label = label
        "The name of the marker"
        self.data = track_data
        "The actual marker data"

    @property
    def X(self) -> np.ndarray:
        """
        Convenience property that returns or sets the
        X component of the marker position.
        """
        return self.data[:, 0]

    @property
    def Y(self) -> np.ndarray:
        """
        Convenience property that returns or sets the
        Y component of the marker position.
        """
        return self.data[:, 1]

    @property
    def Z(self) -> np.ndarray:
        """
        Convenience property that returns or sets the
        Z component of the marker position.
        """
        return self.data[:, 2]

    @X.setter
    def X(self, value) -> None:
        self.data[:, 0] = value

    @Y.setter
    def Y(self, value) -> None:
        self.data[:, 1] = value

    @Z.setter
    def Z(self, value) -> None:
        self.data[:, 2] = value

    @property
    def nFrames(self) -> int:
        """
        Returns:
            int: number of frames in the track
        """
        return self.data.shape[0]

    @property
    def _segments(self):
        maskedTrackData = np.ma.masked_invalid(self.data)
        return np.ma.clump_unmasked(maskedTrackData.T[0])

    @staticmethod
    def _build(stream, nFrames: int) -> "MarkerTrack":
        trackData = np.empty(nFrames, dtype=TrackType.btype)
        trackData[:] = np.NaN

        label = BTSString.bread(stream, 256)
        nSegments = i32.bread(stream)
        i32.skip(stream)
        segmentData = SegmentData.bread(stream, nSegments)
        for startFrame, nFrames in segmentData:
            dat = TrackType.bread(stream, nFrames)
            trackData[startFrame : startFrame + nFrames] = dat
        return MarkerTrack(label, trackData)

    def _write(self, file) -> None:
        # label
        BTSString.bwrite(file, 256, self.label)

        segments = self._segments

        # nSegments
        i32.bwrite(file, len(segments))

        # padding
        i32.bpad(file, 1)

        for segment in segments:
            # startFrame
            i32.bwrite(file, np.array(segment.start))
            # nFrames
            i32.bwrite(file, np.array(segment.stop - segment.start))

        for segment in segments:
            # trackData
            TrackType.bwrite(file, self.data[segment])

    @property
    def nBytes(self) -> int:
        """
        Returns:
            int: size of the track in bytes
        """
        base = 256 + 4 + 4
        for segment in self._segments:
            base += 4 + 4 + (segment.stop - segment.start) * TrackType.btype.itemsize
        return base

    def __repr__(self) -> str:
        return f"Track(label={self.label}, nFrames={self.nFrames})"

    def __eq__(self, other):
        return self.label == other.label and np.all(self.data == other.data)


class Data3D(Block):
    type = BlockType.data3D

    def __init__(
        self,
        frequency: float,
        nFrames: int,
        volume: np.ndarray,
        rotationMatrix: np.ndarray,
        translationVector: np.ndarray,
        startTime: float = 0.0,
        flag: Flags = Flags.rawData,
        format: Data3dBlockFormat = Data3dBlockFormat.byTrack,
    ) -> None:
        """A data block that contains marker tracks.

        Args:
            frequency (float): data frequency in Hz
            nFrames (int): number of frames in the data block
            volume (Union[Volume,np.ndarray]): acquisition volume
            rotationMatrix (np.ndarray): rotation matrix
            translationVector (np.ndarray): translation vector
            startTime (float, optional): Acquisition start time. Defaults to 0.0.
            flag (Flags, optional): Data 3D block flags. Defaults to Flags.rawData.
            format (Data3dBlockFormat, optional): Data3D format.
             Defaults to Data3dBlockFormat.byTrack.

        Raises:
            ValueError: the volume is not an array of 3 floats
            ValueError: the rotation matrix is not an array of 3x3 floats
            ValueError: the translation vector is not an array of 3 floats
        """
        super().__init__()
        self.format = format
        self.frequency = frequency
        self.startTime = startTime

        if not (
            isinstance(rotationMatrix, np.ndarray)
            and rotationMatrix.shape == MAT3X3F.btype.shape
        ):
            raise ValueError(
                f"rotationMatrix must be a numpy array of shape {MAT3X3F.btype.shape}"
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

        self.flag = flag
        self.nFrames = nFrames

        self._tracks = []

    def add_track(self, track: MarkerTrack) -> None:
        """Adds a track to the data block

        Args:
            track (MarkerTrack): track to add

        Raises:
            TypeError: Track is not of type MarkerTrack
            ValueError: Track has a different number of frames than the data block
        """
        if not isinstance(track, MarkerTrack):
            raise TypeError("Track must be of type Track")
        if track.nFrames != self.nFrames:
            raise ValueError(
                (
                    f"Track with label {track.label} has"
                    f" {track.nFrames} frames, expected {self.nFrames} frames"
                )
            )
        self._tracks.append(track)

    @property
    def tracks(self) -> List[MarkerTrack]:
        """Returns a list of all tracks in the data block

        Returns:
            List[MarkerTrack]: list of all tracks in the data block
        """
        return self._tracks

    @tracks.setter
    def tracks(self, values: Iterable[MarkerTrack]) -> None:
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

    @staticmethod
    def _build(stream, format) -> "Data3D":
        format = Data3dBlockFormat(format)
        nFrames = i32.bread(stream)
        frequency = i32.bread(stream)
        startTime = f32.bread(stream)
        nTracks = u32.bread(stream)
        volume = Volume.bread(stream)
        rotationMatrix = MAT3X3F.bread(stream)
        translationVector = VEC3F.bread(stream)
        flag = Flags(u32.bread(stream))

        d = Data3D(
            frequency,
            nFrames,
            volume,
            rotationMatrix,
            translationVector,
            startTime,
            flag,
            format,
        )

        if format in [Data3dBlockFormat.byTrack, Data3dBlockFormat.byFrame]:
            nLinks = i32.bread(stream)
            i32.skip(stream, 1)
            d.links = LinkType.bread(stream, nLinks)

        if format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            d._tracks = [MarkerTrack._build(stream, nFrames) for _ in range(nTracks)]
        else:
            raise NotImplementedError(f"Data3D format {format} not implemented yet")
        return d

    def __getitem__(self, key: Union[int, str]) -> MarkerTrack:
        if isinstance(key, int):
            return self._tracks[key]
        elif isinstance(key, str):
            try:
                return next(track for track in self._tracks if track.label == key)
            except StopIteration:
                raise KeyError(f"Track with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __iter__(self) -> Iterator[MarkerTrack]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __eq__(self, other) -> bool:
        buff1 = BytesIO()
        buff2 = BytesIO()
        self._write(buff1)
        other._write(buff2)
        return buff1.getvalue() == buff2.getvalue()

    def __contains__(self, value: Union[MarkerTrack, str]) -> bool:
        if isinstance(value, MarkerTrack):
            return value in self._tracks
        elif isinstance(value, str):
            return any(track.label == value for track in self._tracks)
        raise TypeError(f"Invalid value type {type(value)}")

    @property
    def nTracks(self) -> int:
        """Number of tracks in the data block

        Returns:
            int: number of tracks
        """
        return len(self._tracks)

    def _write(self, file: BinaryIO) -> None:
        if self.format not in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )

        # nFrames
        i32.bwrite(file, self.nFrames)
        # frequency
        i32.bwrite(file, self.frequency)
        # startTime
        f32.bwrite(file, self.startTime)
        # nTracks
        u32.bwrite(file, len(self._tracks))

        # volume
        Volume.bwrite(file, self.volume)
        # rotationMatrix
        MAT3X3F.bwrite(file, self.rotationMatrix)
        # translationVector
        VEC3F.bwrite(file, self.translationVector)
        # flags
        u32.bwrite(file, self.flag.value)

        if self.format in [
            Data3dBlockFormat.byFrame,
            Data3dBlockFormat.byTrack,
        ]:
            links = self.links if hasattr(self, "links") else []
            nLinks = len(links)

            # nLinks
            i32.bwrite(file, nLinks)
            # padding
            i32.bpad(file)
            # links
            LinkType.bwrite(file, links)

        for track in self._tracks:
            track._write(file)

    @property
    def nBytes(self) -> int:
        base = (
            4  # nFrames
            + 4  # frequency
            + 4  # startTime
            + 4  # nTracks
            + Volume.btype.itemsize  # volume
            + MAT3X3F.btype.itemsize  # rotationMatrix
            + VEC3F.btype.itemsize  # translationVector
            + 4  # flags
        )

        if self.format in [
            Data3dBlockFormat.byFrame,
            Data3dBlockFormat.byTrack,
        ]:
            links_size = (
                4
                + 4
                + (
                    LinkType.btype.itemsize * len(self.links)
                    if hasattr(self, "links")
                    else 0
                )
            )
            base += links_size

        for track in self._tracks:
            base += track.nBytes

        return base

    def __repr__(self) -> str:
        return (
            f"<Data3D "
            f"format={self.format.name}, "
            f"nFrames={self.nFrames}, "
            f"frequency={self.frequency}, "
            f"startTime={self.startTime}, "
            f"nTracks={self.nTracks}, "
            f"tracks={[i.label for i in self._tracks]}>"
        )
