__doc__ = """
Force platform calibration data module.
"""

from enum import Enum
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType
from basictdf.tdfTypes import VEC2F, BTSString, TdfType, i16, i32

# type that stores the 4 vertices of a force platform as a 4x3 float matrix
ForcePlatformVertices = TdfType(np.dtype("(4,3)<f4"))


class ForcePlatformCalibrationBlockFormat(Enum):
    unknownFormat = 0
    ISSFormat = 1  # TDF_CALPLAT_FORMAT_ISS
    GRPFormat = 2  # TDF_CALPLAT_FORMAT_GRP? (not documented)


class ForcePlatformInfo:
    """
    Class that stores the calibration data of a force platform.
    """

    def __init__(self, label, size, position) -> None:
        self.label = label
        "Force platform label"
        self.size = size
        "Size in meters (width, length)"
        self.position = position
        "Position of the 4 vertices in x,y,z coordinates"

    @staticmethod
    def _build(stream) -> "ForcePlatformInfo":
        label = BTSString.bread(stream, 256)  # Docs say 32, but it's actually 256
        size = VEC2F.bread(stream)
        position = ForcePlatformVertices.bread(stream)
        BTSString.bread(stream, 256)  # Undocumented padding

        return ForcePlatformInfo(label, size, position)

    def _write(self, stream) -> None:
        BTSString.bwrite(stream, 256, self.label)
        VEC2F.bwrite(stream, self.size)
        ForcePlatformVertices.bwrite(stream, self.position)
        BTSString.bwrite(stream, 256, "")  # Undocumented padding

    nBytes = 256 + (4 * 2) + (4 * 3 * 4) + 256

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ForcePlatformInfo):
            return False
        return (
            self.label == o.label
            and np.allclose(self.size, o.size)
            and np.allclose(self.position, o.position)
        )

    def __repr__(self) -> str:
        # return (
        #     f"ForcePlatform(label={self.label}, size={self.size},"
        #     " position={self.position})"
        # )
        return (
            "<ForcePlatformInfo "
            f"label={self.label}, size={self.size}, position={self.position}>"
        )


class ForcePlatformsCalibrationDataBlock(Block):
    """
    Block that stores the calibration data of a list of force platforms.
    Each platform is stored as a ForcePlatform object, along with a channel
    number. The channel number links the logical channel of the platform to the
    physical channel of the data acquisition system.
    """

    type = BlockType.forcePlatformsCalibrationData

    def __init__(
        self,
        platforms=None,
        format=ForcePlatformCalibrationBlockFormat.GRPFormat,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._platforms: List[ForcePlatformInfo] = platforms or []
        self._platformMap = []
        self.format = format

    @staticmethod
    def _build(stream, format) -> "ForcePlatformsCalibrationDataBlock":
        format = ForcePlatformCalibrationBlockFormat(format)
        if format != ForcePlatformCalibrationBlockFormat.GRPFormat:
            raise NotImplementedError(
                f"Platform calibration format {format} not implemented yet"
            )
        nPlats = i32.bread(stream)
        i32.skip(stream)  # padding
        platMap = i16.bread(stream, n=nPlats)
        block = ForcePlatformsCalibrationDataBlock(format=format)
        for nPlat in range(nPlats):
            platform = ForcePlatformInfo._build(stream)
            block.add_platform(platform, channel=platMap[nPlat])

        return block

    def _write(
        self, file, format=ForcePlatformCalibrationBlockFormat.GRPFormat
    ) -> None:
        if format != ForcePlatformCalibrationBlockFormat.GRPFormat:
            raise NotImplementedError(
                f"Platform calibration format {format} not implemented yet"
            )

        # nPlats
        i32.bwrite(file, len(self))

        # padding
        i32.bpad(file, 1)

        # platMap
        i16.bwrite(file, self._platformMap)

        # platforms
        for platform in self._platforms:
            platform._write(file)

    def add_platform(self, platform: ForcePlatformInfo, channel: int = None):
        """
        Adds a platform to the list of platforms. Optionally, a channel can be
        specified. If no channel is specified, the next available channel is
        used.
        """
        if not isinstance(platform, ForcePlatformInfo):
            raise TypeError("platform must be of type ForcePlatform")

        if channel is None:
            if len(self._platformMap) == 0:
                next_channel = 0
            else:
                next_channel = max(self._platformMap) + 1
            self._platformMap.append(next_channel)
        else:
            if channel in self._platformMap:
                raise ValueError(f"channel {channel} already in use")
            self._platformMap.append(channel)
        self._platforms.append(platform)

    def remove_platform(self, plat):
        """
        Removes a platform from the list of platforms. The platform can be
        specified either by index or by ForcePlatform object.
        """
        if isinstance(plat, ForcePlatformInfo):
            if plat in self._platforms:
                index = self._platforms.index(plat)
            else:
                raise ValueError(f"platform {plat} not in list")
        elif isinstance(plat, int):
            if plat >= len(self._platforms):
                raise ValueError(f"index {plat} out of range")
            else:
                index = plat

        del self._platforms[index]
        del self._platformMap[index]

    def remove_platforms(self, plats):
        """
        Removes a list of platforms from the list of platforms. The platforms
        can be specified either by index or by ForcePlatform object.
        """
        for plat in plats:
            self.remove_platform(plat)

    def add_platforms(self, plats, channels=None):
        """
        Adds a list of platforms to the list of platforms. If a list of
        channels is provided, the platforms will be added to the corresponding
        channels. If no list of channels is provided, the platforms will be
        added to the next available channels.

        """
        if channels:
            for plat, channel in zip(plats, channels):
                self.add_platform(plat, channel)
        else:
            for plat in plats:
                self.add_platform(plat)

    @property
    def platforms(self) -> List[Tuple[int, ForcePlatformInfo]]:
        """
        Returns:
            List[Tuple(int, ForcePlatform)]: a list of tuples containing the
            channel and the platform

        Settable:
            Iterable[Sequence[Union[int, ForcePlatform]]]: a list of tuples
            containing the channel and the platform
        """
        return list(zip(self._platformMap, self._platforms))

    @platforms.setter
    def platforms(
        self, channel_plats: Iterable[Sequence[Union[int, ForcePlatformInfo]]]
    ):
        """
        Replaces the current list of platforms with the one provided. The
        input is a list of tuples containing the channel and the platform.
        """
        self._platformMap = []
        self._platforms = []
        for channel, plat in channel_plats:
            self.add_platform(plat, channel)

    @property
    def nBytes(self) -> int:
        return (
            4  # nPlats
            + 4  # padding
            + 2 * len(self._platformMap)
            + sum(platform.nBytes for platform in self._platforms)
        )

    def __getitem__(self, key):
        return self._platforms[key]

    def __iter__(self) -> Iterator[ForcePlatformInfo]:
        return iter(self.platforms)

    def __contains__(self, item: ForcePlatformInfo) -> bool:
        return item in self._platforms

    def __len__(self) -> int:
        return len(self._platforms)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ForcePlatformsCalibrationDataBlock):
            return False
        return (
            self.format == o.format
            and self._platformMap == o._platformMap
            and self._platforms == o._platforms
        )

    def __repr__(self) -> str:
        return (
            f"<ForcePlatformsCalibrationDataBlock "
            f"format={self.format.name}, nPlats={len(self)}>"
        )
