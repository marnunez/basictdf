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


class Track:
    def __init__(self, label, trackData):
        self.label = label
        # self.data = trackData.astype(VEC3F.btype).reshape(-1, 3)
        self.data = np.recarray(trackData.shape, dtype=VEC3F.btype)
        print(self.data.shape)
        self.data = trackData[:]

    @property
    def _segments(self):
        maskedTrackData = np.ma.masked_invalid(self.data["X"].astype(np.float32))
        return np.ma.clump_unmasked(maskedTrackData)

    def write(self):
        data = b""

        # label
        data += BTSString.write(256, self.label)

        segments = self._segments
        # nSegments
        data += Int32.write(np.array(len(segments)))
        # padding
        data += Int32.pad(1)

        for segment in segments:
            # startFrame
            data += Int32.write(np.array(segment.start))
            # nFrames
            data += Int32.write(np.array(segment.stop - segment.start))
            # trackData
            data += VEC3F.write(self.data[segment])

        return data

    def __repr__(self):
        return "Track(label={})".format(self.label)


class Data3D(Block):
    def __init__(self, entry, block_data):
        super().__init__(entry, block_data)
        self.format = Data3dBlockFormat(entry["format"])
        self.nFrames = Int32.read(self.data, 1)[0]
        self.frequency = Int32.read(self.data, 1)[0]
        self.startTime = Float32.read(self.data, 1)[0]
        self.nTracks = Uint32.read(self.data, 1)[0]
        self.volume = Volume.read(self.data, 1)[0]
        self.rotationMatrix = Matrix.read(self.data, 1)[0]
        self.translationVector = VEC3F.read(self.data, 1)[0]
        self.flags = Flags(Uint32.read(self.data, 1)[0])

        if self.format in [Data3dBlockFormat.byTrack, Data3dBlockFormat.byFrame]:
            self.nLinks = Int32.read(self.data, 1)[0]
            self.data.seek(4, 1)
            self.links = LinkType.read(self.data, self.nLinks)

        self.tracks = []
        if self.format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            for _ in range(self.nTracks):
                label = BTSString.read(256, self.data.read(256))
                nSegments = Int32.read(self.data, 1)[0]
                Int32.skip(self.data, 1)
                segmentData = SegmentData.read(self.data, nSegments)
                trackData = np.empty(self.nFrames, dtype=VEC3F.btype)
                trackData[:] = np.NaN
                for startFrame, nFrames in segmentData:
                    trackData[startFrame : startFrame + nFrames] = VEC3F.read(
                        self.data, nFrames
                    )
                self.tracks.append(Track(label, trackData))
        else:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )

    def write(self):
        data = b""

        # nFrames
        data += Int32.write(self.nFrames)
        # frequency
        data += Int32.write(self.frequency)
        # startTime
        data += Float32.write(self.startTime)
        # nTracks
        data += Uint32.write(np.array(len(self.tracks)))
        # volume
        data += Volume.write(self.volume)
        # rotationMatrix
        data += Matrix.write(self.rotationMatrix)
        # translationVector
        data += VEC3F.write(self.translationVector)
        # flags
        data += Uint32.write(np.array(self.flags.value))

        if self.nLinks and self.links.size:
            # nLinks
            data += Int32.write(np.array(len(self.links)))
            # padding
            data += Int32.pad(1)
            # links
            data += LinkType.write(self.links)

        if self.format in [
            Data3dBlockFormat.byTrack,
            Data3dBlockFormat.byTrackWithoutLinks,
        ]:
            for track in self.tracks:
                data += track.write()

        else:
            raise NotImplementedError(
                f"Data3D format {self.format} not implemented yet"
            )
        return data

    def __repr__(self):
        return f"<Data3D: {self.nFrames} frames, {self.nTracks} tracks, tracks={self.tracks}>"
