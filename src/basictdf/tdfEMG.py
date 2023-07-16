__doc__ = """

EMG data module.

This module contains the classes to represent EMG data in a TDF file.

Example:

```python
from basictdf import Tdf, EMG, EMGTrack
import numpy as np

# Create an empty EMG block
emg = EMG(frequency=1000, nSamples=1000)

# Create a bogus EMG track with random data
emgTrack = EMGTrack("emg", np.random.rand(1000))

# Add it to the EMG block
emg.addSignal(emgTrack)

# Write it to a new TDF file
with Tdf.new("my_file.tdf").allow_write() as tdf:
    tdf.emg = emg

# This works too
tdf = Tdf.new("my_file.tdf")
with tdf.allow_write() as tdf:
    tdf.emg = emg

```

"""


from enum import Enum
from typing import Iterator, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType, Sized, BuildWriteable
from basictdf.tdfTypes import BTSString, TdfType, f32, i16, i32

SegmentData = TdfType(np.dtype([("startFrame", "<i4"), ("nFrames", "<i4")]))


class EMGBlockFormat(Enum):
    unknownFormat = 0
    byTrack = 1
    byFrame = 2


class EMGTrack(Sized, BuildWriteable):
    def __init__(self, label: str, trackData: np.ndarray) -> None:
        self.label = label
        self.data = trackData

    @property
    def nSamples(self) -> int:
        """
        Returns:
            int: number of samples of the track
        """
        return self.data.shape[0]

    @property
    def _segments(self):
        maskedTrackData = np.ma.masked_invalid(self.data)
        return np.ma.clump_unmasked(maskedTrackData.T)

    @staticmethod
    def _build(stream, nSamples) -> "EMGTrack":
        label = BTSString.bread(stream, 256)
        nSegments = i32.bread(stream)
        i32.skip(stream)  # padding
        segmentData = SegmentData.bread(stream, nSegments)
        trackData = np.empty(nSamples, dtype="<f4")
        trackData[:] = np.nan
        for startFrame, nFrames in segmentData:
            trackData[startFrame : startFrame + nFrames] = f32.bread(stream, nFrames)
        return EMGTrack(label, trackData)

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
            i32.bwrite(file, segment.start)
            # nFrames
            i32.bwrite(file, segment.stop - segment.start)

        for segment in segments:
            # data
            f32.bwrite(file, self.data[segment])

    @property
    def nBytes(self):
        base = 256 + 4 + 4
        for segment in self._segments:
            base += 4 + 4 + (segment.stop - segment.start) * f32.btype.itemsize
        return base

    def __eq__(self, other):
        return self.label == other.label and np.all(self.data == other.data)

    def __repr__(self) -> str:
        return (
            f"EMGTrack(label={self.label}, nSamples={self.nSamples},"
            f"segments={len(self._segments)})"
        )


class EMG(Block):
    """Electromyography data block"""

    type = BlockType.electromyographicData

    def __init__(
        self, frequency, nSamples, startTime=0.0, format=EMGBlockFormat.byTrack
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.startTime = startTime
        self.nSamples = nSamples
        self._signals = []
        self._emgMap = []
        self.format = format

    @staticmethod
    def _build(stream, format) -> "EMG":
        format = EMGBlockFormat(format)
        nSignals = i32.bread(stream)
        frequency = i32.bread(stream)
        startTime = f32.bread(stream)
        nSamples = i32.bread(stream) + 49  # Why 49??? Whyyyy????
        emgMap = i16.bread(stream, n=nSignals)

        d = EMG(frequency, nSamples, startTime, format)
        if format == EMGBlockFormat.byTrack:
            for n in range(nSignals):
                emgSignal = EMGTrack._build(stream, nSamples)
                d.addSignal(emgSignal, channel=emgMap[n])
        else:
            raise NotImplementedError(f"EMG format {format} not implemented yet")
        return d

    def _write(self, file) -> None:
        if self.format != EMGBlockFormat.byTrack:
            raise NotImplementedError(f"EMG format {self.format} not implemented yet")

        # nSignals
        i32.bwrite(file, len(self._signals))

        # frequency
        i32.bwrite(file, self.frequency)

        # startTime
        f32.bwrite(file, self.startTime)

        # nSamples
        i32.bwrite(file, self.nSamples - 49)  # That 49 again

        # emgMap
        i16.bwrite(file, self._emgMap)

        # signals
        for signal in self._signals:
            signal._write(file)

    def __getitem__(self, key) -> EMGTrack:
        if isinstance(key, int):
            return self._signals[key]
        elif isinstance(key, str):
            try:
                return next(signal for signal in self._signals if signal.label == key)
            except StopIteration:
                raise KeyError(f"EMG signal with label {key} not found")
        raise TypeError(f"Invalid key type {type(key)}")

    def __contains__(self, value: Union[EMGTrack, str]) -> bool:
        if isinstance(value, str):
            return any(signal.label == value for signal in self._signals)
        elif isinstance(value, EMGTrack):
            return value in self._signals
        raise TypeError(f"Invalid value type {type(value)}")

    def __iter__(self) -> Iterator[EMGTrack]:
        return iter(self._signals)

    def __len__(self) -> int:
        return len(self._signals)

    def __eq__(self, other) -> bool:
        if not isinstance(other, EMG):
            return False
        return (
            self.frequency == other.frequency
            and self.startTime == other.startTime
            and self.nSamples == other.nSamples
            and all(s1 == s2 for s1, s2 in zip(self._signals, other._signals))
        )

    def addSignal(self, signal: EMGTrack, channel=None) -> None:
        """
        Adds a signal to the EMG block. If the channel is not specified,
        it is set to the next one  available
        """
        if not isinstance(signal, EMGTrack):
            raise TypeError(f"Can only add EMGTrack objects, got {type(signal)}")
        if signal.nSamples != self.nSamples:
            raise ValueError(
                (
                    f"EMGTrack with label {signal.label} has {signal.nSamples} "
                    f"samples, expected {self.nSamples}"
                )
            )

        if channel is None:
            if len(self._emgMap) == 0:
                next_channel = 0
            else:
                next_channel = max(self._emgMap) + 1
            self._emgMap.append(next_channel)
        else:
            if channel in self._emgMap:
                raise ValueError(f"Channel {channel} already in use")
            self._emgMap.append(channel)
        self._signals.append(signal)

    def removeSignal(self, label: str) -> None:
        """
        Removes a signal specified by its label from the EMG block
        """
        try:
            pos = next(i for i, v in enumerate(self._signals) if v == label)
        except StopIteration:
            raise KeyError(f"EMG signal with label {label} not found")

        del self._signals[pos]
        del self._emgMap[pos]

    @property
    def nBytes(self) -> int:
        base = 4 + 4 + 4 + 2 * len(self._signals) + 4
        for signal in self._signals:
            base += signal.nBytes
        return base

    @property
    def nSignals(self) -> int:
        return len(self._signals)

    def __repr__(self) -> str:
        return (
            "<EMGBlock"
            f" format={self.format.name}"
            f" frequency={self.frequency}"
            f" nSamples={self.nSamples}"
            f" nSignals={self.nSignals}"
            f" startTime={self.startTime}"
            ">"
        )
