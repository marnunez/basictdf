__doc__ = """
Force platform data module.
"""

from enum import Enum
from basictdf.tdfBlock import Block, BlockType, Sized, BuildWriteable
from basictdf.tdfTypes import i32, u16, SegmentData, TdfType, f32

import numpy as np

PlatDataType = TdfType(
    np.dtype([("application_point", "2<f4"), ("force", "3<f4"), ("torque", "<f4")])
)


class ForcePlatformBlockFormat(Enum):
    unknownFormat = 0
    byTrackISSFormat = 1  # TDF_DATAPLAT_FORMAT_BYTRACK_ISS
    byFrameISSFormat = 2  # TDF_DATAPLAT_FORMAT_BYFRAME_ISS
    byTrackISSWithLabelsFormat = 3  # TDF_DATAPLAT_FORMAT_BYTRACK_ISS_WL
    byFrameISSWithLabelsFormat = 4  # TDF_DATAPLAT_FORMAT_BYFRAME_ISS_WL
    byTrackDoubleFormat = 5  # TDF_DATAPLAT_FORMAT_BYTRACK_DBL
    byFrameDoubleFormat = 6  # TDF_DATAPLAT_FORMAT_BYFRAME_DBL
    byTrackDoubleWithLabelsFormat = 7  # TDF_DATAPLAT_FORMAT_BYTRACK_WL_DBL
    byFrameDoubleWithLabelsFormat = 8  # TDF_DATAPLAT_FORMAT_BYFRAME_WL_DBL
    byTrackISSWithVelocityFormat = 9  # TDF_DATAPLAT_FORMAT_BYTRACK_ISS_VEL
    byFrameISSWithVelocityFormat = 10  # TDF_DATAPLAT_FORMAT_BYFRAME_ISS_VEL
    byTrackISSWithLabelsAndVelocityFormat = 11  # TDF_DATAPLAT_FORMAT_BYTRACK_WL_ISS_VEL
    byFrameISSWithLabelsAndVelocityFormat = 12  # TDF_DATAPLAT_FORMAT_BYFRAME_WL_ISS_VEL
    byTrackDoubleWithVelocityFormat = 13  # TDF_DATAPLAT_FORMAT_BYTRACK_DBL_VEL
    byFrameDoubleWithVelocityFormat = 14  # TDF_DATAPLAT_FORMAT_BYFRAME_DBL_VEL


class ForcePlatformData(Sized, BuildWriteable):
    """
    Class that stores the data of a force platform,
    such as force, torque and application point.
    """

    def __init__(
        self,
        label: str,
        application_point: np.array,
        force: np.array,
        torque: np.array,
    ) -> None:
        self.label = label
        "Force platform label"
        self.application_point = application_point
        "Position of the application point in x,y coordinates"
        self.force = force
        "Force in x,y,z coordinates"
        self.torque = torque
        "Torque in z axis"

    @staticmethod
    def _build(
        stream, format: ForcePlatformBlockFormat, n_frames: int
    ) -> "ForcePlatformData":
        n_segments = i32.bread(stream)
        i32.skip(stream)  # padding
        segment_data = SegmentData.bread(stream, n_segments)
        data = np.empty(n_frames, dtype=PlatDataType.btype)
        for start_frame, n_frames in segment_data:
            dat = PlatDataType.bread(stream, n_frames)
            data[start_frame : start_frame + n_frames] = dat
        application_point = data["application_point"]
        force = data["force"]
        torque = data["torque"]
        return ForcePlatformData("", application_point, force, torque)

    @property
    def _segments(self):
        # Wherever application_point is masked, force and torque are also masked
        maskedData = np.ma.masked_invalid(self.application_point)
        return np.ma.clump_unmasked(maskedData.T[0])

    def _get_segment_data(self, segment):
        start, stop = segment.start, segment.stop
        app_point = self.application_point[start:stop]
        force = self.force[start:stop]
        torque = self.torque[start:stop]
        return np.rec.fromarrays([app_point, force, torque], dtype=PlatDataType.btype)

    def _write(self, stream, format) -> None:
        if format != ForcePlatformBlockFormat.byTrackISSFormat:
            raise NotImplementedError(
                f"ForcePlatformDataBlock format {format} not implemented"
            )
        segments = self._segments
        i32.bwrite(stream, len(segments))
        i32.bpad(stream)
        for segment in segments:
            # startFrame
            i32.bwrite(stream, np.array(segment.start))
            # nFrames
            i32.bwrite(stream, np.array(segment.stop - segment.start))

        for segment in segments:
            PlatDataType.bwrite(stream, self._get_segment_data(segment))

    @property
    def nBytes(self) -> int:
        "Size in bytes of the platform data"
        nSegments = len(self._segments)

        base = 4 + 4 + (4 + 4) * nSegments

        for segment in self._segments:
            base += PlatDataType.btype.itemsize * (segment.stop - segment.start)

        return base

    def __repr__(self):
        return (
            "<ForcePlatformData "
            f"application_point={self.application_point}, "
            f"force={self.force}, "
            f"torque={self.torque}>"
        )

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ForcePlatformData)
            and np.allclose(self.application_point, __value.application_point)
            and np.allclose(self.force, __value.force)
            and np.allclose(self.torque, __value.torque)
        )


class ForcePlatformsDataBlock(Block):
    """
    Block that stores the data of a force platform,
    such as force, torque and application point.
    """

    type = BlockType.forcePlatformsData

    def __init__(
        self,
        start_time,
        frequency,
        n_frames,
        format=ForcePlatformBlockFormat.byTrackISSFormat,
    ) -> None:
        super().__init__()
        self.format = format
        self._plat_map = []
        self._platforms = []

        self.start_time = start_time
        "Start time of the capture"
        self.frequency = frequency
        "Frequency of the capture in Hz"
        self.n_frames = n_frames
        "Number of frames captured"

    @staticmethod
    def _build(stream, format) -> "ForcePlatformsDataBlock":
        format = ForcePlatformBlockFormat(format)
        if format != ForcePlatformBlockFormat.byTrackISSFormat:
            raise NotImplementedError(
                f"ForcePlatformDataBlock format {format} not implemented"
            )
        n_plats = i32.bread(stream)
        frequency = i32.bread(stream)
        start_time = f32.bread(stream)
        n_frames = i32.bread(stream)
        plat_map = u16.bread(stream, n_plats)
        platforms = [
            ForcePlatformData._build(stream, format, n_frames) for _ in range(n_plats)
        ]
        block = ForcePlatformsDataBlock(start_time, frequency, n_frames)
        block._plat_map = plat_map
        block._platforms = platforms

        return block

    def _write(self, stream) -> None:
        i32.bwrite(stream, len(self._platforms))
        i32.bwrite(stream, self.frequency)
        f32.bwrite(stream, self.start_time)
        i32.bwrite(stream, self.n_frames)
        u16.bwrite(stream, self._plat_map)
        for plat in self._platforms:
            plat._write(stream, self.format)

    def __iter__(self):
        """
        Iterates over the platforms in the block.
        Returns a tuple of (channel, platform)
        """
        return iter(zip(self._plat_map, self._platforms))

    def add_platform(self, platform: ForcePlatformData, channel: int = None) -> None:
        """
        Adds a platform to the block. If channel is specified and it's not already
        in use, the platform is added to that channel. Otherwise, the platform is
        added to the next available channel.
        """
        if channel is None:
            channel = len(self._platforms)
        if channel in self._plat_map:
            raise ValueError(f"Channel {channel} already in use")
        self._plat_map.append(channel)
        self._platforms.append(platform)

    @property
    def nBytes(self) -> int:
        return (
            4  # nPlatforms
            + 4  # frequency
            + 4  # startTime
            + 4  # nFrames
            + len(self._platforms) * 2  # platMap
            + sum(plat.nBytes for plat in self._platforms)
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ForcePlatformsDataBlock):
            return False
        return (
            self.start_time == o.start_time
            and self.frequency == o.frequency
            and self.n_frames == o.n_frames
            and np.array_equal(self._plat_map, o._plat_map)
            and all(plat == oplat for plat, oplat in zip(self._platforms, o._platforms))
        )

    def __repr__(self) -> str:
        return (
            "<ForcePlatformsDataBlock "
            f"format={self.format.name}, "
            f"nPlatforms={len(self._platforms)}, "
            f"frequency={self.frequency}, "
            f"startTime={self.start_time}>"
        )
