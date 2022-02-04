from io import BytesIO
from basictdf.tdfBlock import Block, BlockType
from enum import Enum
from basictdf.tdfTypes import (
    BTSString,
    Int32,
    Uint32,
    Volume,
    VEC3F,
    Matrix,
    Float32,
    Type,
)
import numpy as np
from basictdf.tdfUtils import is_iterable


class Data3dBlockFormat(Enum):
    unknownFormat = 0
    byTrack = 1
    byTrackWithoutLinks = 2
    byFrame = 3
    byFrameWithoutLinks = 4


class Flags(Enum):
    rawData = 0
    filtered = 1


LinkType = Type(np.dtype([("Track1", "<u4"), ("Track2", "<u4")]))
SegmentData = Type(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))

TrackType = Type(np.dtype("<3f4"))


class Track:
    def __init__(self, label, trackData):
        self.label = label
        self.data = trackData

    @property
    def X(self):
        return self.data[:, 0]

    @property
    def Y(self):
        return self.data[:, 1]

    @property
    def Z(self):
        return self.data[:, 2]

    @X.setter
    def X(self, value):
        self.data[:, 0] = value

    @Y.setter
    def Y(self, value):
        self.data[:, 1] = value

    @Z.setter
    def Z(self, value):
        self.data[:, 2] = value

    @property
    def nFrames(self):
        return self.data.shape[0]

    @property
    def _segments(self):
        maskedTrackData = np.ma.masked_invalid(self.data)
        return np.ma.clump_unmasked(maskedTrackData.T[0])

    @staticmethod
    def build(stream, nFrames):
        label = BTSString.bread(stream, 256)
        nSegments = Int32.bread(stream)
        Int32.skip(stream)
        segmentData = SegmentData.bread(stream, nSegments)
        trackData = np.empty(nFrames, dtype=TrackType.btype)
        trackData[:] = np.NaN
        for startFrame, nFrames in segmentData:
            dat = TrackType.bread(stream, nFrames)
            trackData[startFrame : startFrame + nFrames] = dat
        return Track(label, trackData)

    def write(self, file):

        # label
        BTSString.bwrite(file, 256, self.label)

        segments = self._segments

        # nSegments
        Int32.bwrite(file, len(segments))

        # padding
        Int32.bpad(file, 1)

        for segment in segments:
            # startFrame
            Int32.bwrite(file, np.array(segment.start))
            # nFrames
            Int32.bwrite(file, np.array(segment.stop - segment.start))
            # trackData
            TrackType.bwrite(file, self.data[segment])

    @property
    def nBytes(self):
        base = 256 + 4 + 4
        for segment in self._segments:
            base += 4 + 4 + (segment.stop - segment.start) * TrackType.btype.itemsize
        return base

    def __repr__(self):
        return f"Track(label={self.label}, nFrames={self.nFrames})"

    def __eq__(self, other):
        return self.label == other.label and np.all(self.data == other.data)


class Data3D(Block):
    def __init__(
        self,
        frequency,
        nFrames,
        volume,
        rotationMatrix,
        translationVector,
        startTime=0.0,
        flag=Flags.rawData,
        format=Data3dBlockFormat.byTrack,
    ):
        super().__init__(BlockType.data3D)
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

        self.flag = flag
        self.nFrames = nFrames

        self._tracks = []

    def add_track(self, track: Track):
        if not isinstance(track, Track):
            raise TypeError(f"Track must be of type Track")
        if track.nFrames != self.nFrames:
            raise ValueError(f"Track must have the same number of frames as the Data3D")
        self._tracks.append(track)

    @property
    def tracks(self):
        return self._tracks

    @tracks.setter
    def tracks(self, values):
        oldTracks = self._tracks
        self._tracks = []
        try:
            for value in values:
                self.add_track(value)
        except Exception:
            self._tracks = oldTracks
            raise

    @staticmethod
    def build(stream, format):
        format = Data3dBlockFormat(format)
        nFrames = Int32.bread(stream)
        frequency = Int32.bread(stream)
        startTime = Float32.bread(stream)
        nTracks = Uint32.bread(stream)
        volume = Volume.bread(stream)
        rotationMatrix = Matrix.bread(stream)
        translationVector = VEC3F.bread(stream)
        flag = Flags(Uint32.bread(stream))

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
            nLinks = Int32.bread(stream)
            Int32.skip(stream, 1)
            d.links = LinkType.bread(stream, nLinks)

        if format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            d._tracks = [Track.build(stream, nFrames) for _ in range(nTracks)]
        else:
            raise NotImplementedError(f"Data3D format {format} not implemented yet")
        return d

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._tracks[key]
        elif isinstance(key, str):
            try:
                return next(track for track in self._tracks if track.label == key)
            except StopIteration:
                raise KeyError(f"Track with label {key} not found")
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
        # data = b""

        # nFrames
        Int32.bwrite(file, self.nFrames)
        # frequency
        Int32.bwrite(file, self.frequency)
        # startTime
        Float32.bwrite(file, self.startTime)
        # nTracks
        Uint32.bwrite(file, np.array(len(self._tracks)))

        # volume
        Volume.bwrite(file, self.volume)
        # rotationMatrix
        Matrix.bwrite(file, self.rotationMatrix)
        # translationVector
        VEC3F.bwrite(file, self.translationVector)
        # flags
        Uint32.bwrite(file, np.array(self.flag.value))

        if self.format in [
            Data3dBlockFormat.byFrame,
            Data3dBlockFormat.byTrack,
        ]:

            links = self.links if hasattr(self, "links") else []
            nLinks = len(links)

            # nLinks
            Int32.bwrite(file, nLinks)
            # padding
            Int32.bpad(file)
            # links
            LinkType.bwrite(file, links)

        if self.format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            for track in self._tracks:
                track.write(file)

        else:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )

    def __repr__(self):
        return f"<Data3D: {self.nFrames} frames, {self.frequency} Hz, {self.nTracks} tracks, tracks={[i.label for i in self._tracks]}>"
