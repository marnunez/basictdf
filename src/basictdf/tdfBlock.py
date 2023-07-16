from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import IO, Optional

__all__ = ["Block", "BlockType"]
__doc__ = "Block and block type classes."


class BlockType(Enum):
    "All posible block types"
    unusedSlot = 0  # TDF_DATABLOCK_NOBLOCK
    notDefined = 1  # TDF_DATABLOCK_NOTYPE
    calibrationData = 2  # TDF_DATABLOCK_CALIB
    calibrationData2D = 3  # TDF_DATABLOCK_DATA2D4C
    data2D = 4  # TDF_DATABLOCK_DATA2D
    data3D = 5  # TDF_DATABLOCK_DATA3D
    opticalSystemConfiguration = 6  # TDF_DATABLOCK_OPTISETUP
    forcePlatformsCalibrationData = 7  # TDF_DATABLOCK_CALPLAT
    forcePlatformsCalibrationData2D = 8  # TDF_DATABLOCK_DATA2D4P
    forcePlatformsData = 9  # TDF_DATABLOCK_DATAPLAT
    anthropometricData = 10  # TDF_DATABLOCK_ANTHROPO
    electromyographicData = 11  # TDF_DATABLOCK_DATAEMG
    forceAndTorqueData = 12  # TDF_DATABLOCK_FORCE3D
    volumetricData = 13  # TDF_DATABLOCK_VOLUME
    analogData = 14  # TDF_DATABLOCK_GENPURPOSE
    generalCalibrationData = 15  # TDF_DATABLOCK_CALGENPURP
    temporalEventsData = 16  # TDF_DATABLOCK_EVENTS


class Sized(ABC):
    @property
    @abstractmethod
    def nBytes(self) -> int:
        "Size in bytes"
        pass


class BuildWriteable(ABC):
    @classmethod
    @abstractmethod
    def _build(cls, file: IO[bytes], format: int):
        pass

    @abstractmethod
    def _write(self, file: IO[bytes]):
        pass


class Block(Sized, BuildWriteable, ABC):
    """
    A class to represent a TDF block.
    """

    type = BlockType.notDefined

    def __init__(
        self,
        creation_date: Optional[datetime] = None,
        last_modification_date: Optional[datetime] = None,
        last_access_date: Optional[datetime] = None,
    ):
        self.creation_date = (
            creation_date if creation_date is not None else datetime.now()
        )
        self.last_modification_date = (
            last_modification_date
            if last_modification_date is not None
            else datetime.now()
        )
        self.last_access_date = (
            last_access_date if last_access_date is not None else datetime.now()
        )

    @abstractmethod
    def __iter__(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} type={self.type}>"


class NotImplementedBlock(Block):
    type = BlockType.notDefined

    @classmethod
    def _build(cls, *args, **kwargs):
        raise NotImplementedError(
            f"Buidling blocks of type {cls.type} is not implemented yet"
        )

    def _write(self, *args, **kwargs):
        raise NotImplementedError(
            f"Writing blocks type {self.type} is not implemented yet."
        )


class CalibrationData2D(NotImplementedBlock):
    type = BlockType.calibrationData2D


class Data2D(NotImplementedBlock):
    type = BlockType.data2D


class ForcePlatformsCalibrationData(NotImplementedBlock):
    type = BlockType.forcePlatformsCalibrationData


class ForcePlatformsCalibrationData2D(NotImplementedBlock):
    type = BlockType.forcePlatformsCalibrationData2D


class ForcePlatformsData(NotImplementedBlock):
    type = BlockType.forcePlatformsData


class AnthropometricData(NotImplementedBlock):
    type = BlockType.anthropometricData


class VolumetricData(NotImplementedBlock):
    type = BlockType.volumetricData


class AnalogData(NotImplementedBlock):
    type = BlockType.analogData


class GeneralCalibrationData(NotImplementedBlock):
    type = BlockType.generalCalibrationData


class UnusedFormat(Enum):
    unusedFormat = 0


class UnusedBlock(Block):
    type = BlockType.unusedSlot
    nBytes = 0
    format = UnusedFormat.unusedFormat

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def _build(*args, **kwargs) -> "UnusedBlock":
        return UnusedBlock()

    def _write(self, *args, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return "<UnusedBlock>"

    def __iter__(self):
        return iter([])

    def __eq__(self, other: "UnusedBlock") -> bool:
        return isinstance(other, UnusedBlock)


class NotDefinedBlock(NotImplementedBlock):
    type = BlockType.notDefined
