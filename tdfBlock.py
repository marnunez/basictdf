from enum import Enum
from io import BytesIO


class BlockType(Enum):
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


class Block:
    def __init__(self, entry, block_data):
        self.type = entry["type"]
        self.creation_date = entry["creation_date"]
        self.last_modification_date = entry["last_modification_date"]
        self.last_access_date = entry["last_access_date"]
        self.block_data = block_data
        self.data = BytesIO(block_data)

    @property
    def size(self):
        return len(self)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.type) + ")"

    def __len__(self):
        return len(self.block_data)


class CalibrationData(Block):
    pass


class CalibrationData2D(Block):
    pass


class Data2D(Block):
    pass


class OpticalSystemConfiguration(Block):
    pass


class ForcePlatformsCalibrationData(Block):
    pass


class ForcePlatformsCalibrationData2D(Block):
    pass


class ForcePlatformsData(Block):
    pass


class AnthropometricData(Block):
    pass


class ElectromyographicData(Block):
    pass


class ForceAndTorqueData(Block):
    pass


class VolumetricData(Block):
    pass


class AnalogData(Block):
    pass


class GeneralCalibrationData(Block):
    pass
