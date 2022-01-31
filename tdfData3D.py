from tdfBlock import Block
from enum import Enum
from tdfUtils import (
    BTSString,
    Int32,
    Uint32,
    Volume,
    VEC3F,
    Matrix,
    Float32,
    Int16,
    Type,
)
import numpy as np


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

    def write(self, file):

        # label
        BTSString.bwrite(file, 256, self.label)

        segments = self._segments

        # nSegments
        Int32.bwrite(file, np.array(len(segments)))

        # padding
        Int32.bpad(file, 1)

        for segment in segments:
            # startFrame
            Int32.bwrite(file, np.array(segment.start))
            # nFrames
            Int32.bwrite(file, np.array(segment.stop - segment.start))
            # trackData
            TrackType.bwrite(file, self.data[segment])

    def __repr__(self):
        return f"Track(label={self.label}, nFrames={self.nFrames})"


class Data3D(Block):
    def __init__(self, entry, block_data):
        super().__init__(entry, block_data)
        self.format = Data3dBlockFormat(entry["format"])
        self.build()

    def build(self):
        self.nFrames = Int32.read(self.data)[0]
        self.frequency = Int32.read(self.data)[0]
        self.startTime = Float32.read(self.data)[0]
        self.nTracks = Uint32.read(self.data)[0]
        self.volume = Volume.read(self.data)[0]
        self.rotationMatrix = Matrix.read(self.data)[0]
        self.translationVector = VEC3F.read(self.data)[0]
        self.flags = Flags(Uint32.read(self.data)[0])

        if self.format in [Data3dBlockFormat.byTrack, Data3dBlockFormat.byFrame]:
            self.nLinks = Int32.read(self.data)[0]
            Int32.skip(self.data, 1)
            self.links = LinkType.read(self.data, self.nLinks)

        self.tracks = []
        if self.format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            for _ in range(self.nTracks):
                label = BTSString.read(256, self.data.read(256))
                nSegments = Int32.read(self.data, 1)[0]
                Int32.skip(self.data)
                segmentData = SegmentData.read(self.data, nSegments)
                trackData = np.empty(self.nFrames, dtype=TrackType.btype)
                trackData[:] = np.NaN
                for startFrame, nFrames in segmentData:
                    trackData[startFrame : startFrame + nFrames] = TrackType.read(
                        self.data, nFrames
                    )
                self.tracks.append(Track(label, trackData))
        else:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.tracks[key]
        elif isinstance(key, str):
            try:
                return next(track for track in self.tracks if track.label == key)
            except StopIteration:
                raise KeyError(f"Track with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __iter__(self):
        return iter(self.tracks)

    def __len__(self):
        return len(self.tracks)

    def write(self, file):
        # data = b""

        # nFrames
        Int32.bwrite(file, self.nFrames)
        # frequency
        Int32.bwrite(file, self.frequency)
        # startTime
        Float32.bwrite(file, self.startTime)
        # nTracks
        Uint32.bwrite(file, np.array(len(self.tracks)))

        # volume
        Volume.bwrite(file, self.volume)
        # rotationMatrix
        Matrix.bwrite(file, self.rotationMatrix)
        # translationVector
        VEC3F.bwrite(file, self.translationVector)
        # flags
        Uint32.bwrite(file, np.array(self.flags.value))

        if self.nLinks and self.links.size:
            # nLinks
            Int32.bwrite(file, np.array(len(self.links)))
            # padding
            Int32.bpad(file)
            # links
            LinkType.bwrite(file, self.links)

        if self.format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            for track in self.tracks:
                track.write(file)

        else:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )

    def __repr__(self):
        return f"<Data3D: {self.nFrames} frames, {self.frequency} Hz, {self.nTracks} tracks, tracks={[i.label for i in self.tracks]}>"
