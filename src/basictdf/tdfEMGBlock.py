from io import BytesIO
from basictdf.tdfBlock import Block, BlockType
from enum import Enum
from basictdf.tdfTypes import BTSString, Int32, Float32, Type, Int16

import numpy as np


SegmentData = Type(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))


class EMGBlockFormat(Enum):
    unknownFormat = 0
    byTrack = 1
    byFrame = 2


class EMGTrack:
    def __init__(self, label, trackData):
        self.label = label
        self.data = trackData

    @property
    def nSamples(self):
        return self.data.shape[0]

    @property
    def _segments(self):
        maskedTrackData = np.ma.masked_invalid(self.data)
        return np.ma.clump_unmasked(maskedTrackData.T)

    @staticmethod
    def build(stream, nSamples):
        label = BTSString.bread(stream, 256)
        nSegments = Int32.bread(stream)
        Int32.skip(stream)  # padding
        segmentData = SegmentData.bread(stream, nSegments)
        trackData = np.empty(nSamples, dtype="<f4")
        trackData[:] = np.nan
        for startFrame, nFrames in segmentData:
            trackData[startFrame : startFrame + nFrames] = Float32.bread(
                stream, nFrames
            )
        return EMGTrack(label, trackData)

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
            Int32.bwrite(file, segment.start)
            # nFrames
            Int32.bwrite(file, segment.stop - segment.start)
            # data
            Float32.bwrite(file, self.data[segment])

    @property
    def nBytes(self):
        base = 256 + 4 + 4
        for segment in self._segments:
            base += 4 + 4 + (segment.stop - segment.start) * Float32.btype.itemsize
        return base

    def __eq__(self, other):
        return self.label == other.label and np.all(self.data == other.data)

    def __repr__(self):
        return f"EMGTrack(label={self.label}, nSamples={self.nSamples},segments={len(self._segments)})"


class EMG(Block):
    def __init__(self, frequency, nSamples, emgMap, startTime=0.0):
        self.frequency = frequency
        self.startTime = startTime
        self.nSamples = nSamples
        self.emgMap = emgMap
        super().__init__(BlockType.electromyographicData)

    @staticmethod
    def build(stream, format):
        format = EMGBlockFormat(format)
        nSignals = Int32.bread(stream)
        frequency = Int32.bread(stream)
        startTime = Float32.bread(stream)
        nSamples = Int32.bread(stream) + 49  # Why 49??? Whyyyy????
        emgMap = Int16.bread(stream, n=nSignals)

        d = EMG(frequency, nSamples, emgMap, startTime)
        if format == EMGBlockFormat.byTrack:
            d._signals = [EMGTrack.build(stream, nSamples) for _ in range(nSignals)]
        else:
            raise NotImplementedError(f"EMG format {format} not implemented yet")
        return d

    def write(self, file):
        if self.format != EMGBlockFormat.byTrack:
            raise NotImplementedError(f"EMG format {self.format} not implemented yet")

        # nSignals
        Int32.bwrite(file, len(self._signals))

        # frequency
        Int32.bwrite(file, self.frequency)

        # startTime
        Float32.bwrite(file, self.startTime)

        # emgMap
        Int16.bwrite(file, self.emgMap)

        # nSamples
        Int32.bwrite(file, self.nSamples - 49)  # That 49 again

        # signals
        for signal in self._signals:
            signal.write(file)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._signals[key]
        elif isinstance(key, str):
            try:
                return next(signal for signal in self._signals if signal.label == key)
            except StopIteration:
                raise KeyError(f"EMG signal with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __iter__(self):
        return iter(self._signals)

    def __len__(self):
        return len(self._signals)

    def __eq__(self, other):
        buff1 = BytesIO()
        buff2 = BytesIO()
        self.write(buff1)
        other.write(buff2)
        return buff1.getvalue() == buff2.getvalue()

    @property
    def nSignals(self):
        return len(self._signals)

    def __repr__(self):
        return f"EMGBlock(frequency={self.frequency}, startTime={self.startTime}, nSamples={self.nSamples}, nSignals={self.nSignals})"
