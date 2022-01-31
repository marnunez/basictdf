from pathlib import Path
import struct
from tdfUtils import BTSDate, BTSString

from tdfEvents import TemporalEventsData
from tdfData3D import Data3D
from tdfBlock import BlockType
from tdfBlock import (
    CalibrationData,
    CalibrationData2D,
    Data2D,
    OpticalSystemConfiguration,
    ForcePlatformsCalibrationData,
    ForcePlatformsCalibrationData2D,
    ForcePlatformsData,
    AnthropometricData,
    ElectromyographicData,
    ForceAndTorqueData,
    VolumetricData,
    AnalogData,
    GeneralCalibrationData,
)


def get_block(block_type):
    if block_type == BlockType.unusedSlot:
        return None
    elif block_type == BlockType.notDefined:
        return None
    elif block_type == BlockType.calibrationData:
        return CalibrationData
    elif block_type == BlockType.calibrationData2D:
        return CalibrationData2D
    elif block_type == BlockType.data2D:
        return Data2D
    elif block_type == BlockType.data3D:
        return Data3D
    elif block_type == BlockType.opticalSystemConfiguration:
        return OpticalSystemConfiguration
    elif block_type == BlockType.forcePlatformsCalibrationData:
        return ForcePlatformsCalibrationData
    elif block_type == BlockType.forcePlatformsCalibrationData2D:
        return ForcePlatformsCalibrationData2D
    elif block_type == BlockType.forcePlatformsData:
        return ForcePlatformsData
    elif block_type == BlockType.anthropometricData:
        return AnthropometricData
    elif block_type == BlockType.electromyographicData:
        return ElectromyographicData
    elif block_type == BlockType.forceAndTorqueData:
        return ForceAndTorqueData
    elif block_type == BlockType.volumetricData:
        return VolumetricData
    elif block_type == BlockType.analogData:
        return AnalogData
    elif block_type == BlockType.generalCalibrationData:
        return GeneralCalibrationData
    elif block_type == BlockType.temporalEventsData:
        return TemporalEventsData
    else:
        raise Exception("Unknown block type")


class Tdf:
    SIGNATURE = b"\x82K`A\xd3\x11\x84\xca`\x00\xb6\xac\x16h\x0c\x08"

    def __init__(self, filename):
        self.filename = Path(filename)
        self.handler = self.filename.open("rb")
        self.signature = self.handler.read(len(self.SIGNATURE))

        if self.signature != self.SIGNATURE:
            raise Exception("Invalid TDF file")

        self.version = struct.unpack("<I", self.handler.read(4))[0]
        self.nEntries = struct.unpack("<i", self.handler.read(4))[0]
        self.handler.seek(8, 1)
        self.creation_date = BTSDate.read(self.handler.read(4))
        self.last_modification_date = BTSDate.read(self.handler.read(4))
        self.last_access_date = BTSDate.read(self.handler.read(4))

        self.handler.seek(4 * 5, 1)
        self.entries = []

        for _ in range(self.nEntries):
            d = {}
            d["type"] = BlockType(struct.unpack("<I", self.handler.read(4))[0])
            d["format"] = struct.unpack("<I", self.handler.read(4))[0]
            d["offset"] = struct.unpack("<i", self.handler.read(4))[0]
            d["size"] = struct.unpack("<i", self.handler.read(4))[0]
            d["creation_date"] = BTSDate.read(self.handler.read(4))
            d["last_modification_date"] = BTSDate.read(self.handler.read(4))
            d["last_access_date"] = BTSDate.read(self.handler.read(4))

            self.handler.seek(4, 1)
            d["comment"] = BTSString.read(256, self.handler.read(256))
            self.entries.append(d)

    def get_block(self, type):
        for entry in self.entries:
            if entry["type"] == type:
                self.handler.seek(entry["offset"])
                data = self.handler.read(entry["size"])
                block_class = get_block(entry["type"])
                return block_class(entry, data)
        raise ValueError("Block not found")

    def add_block(self, block, position=-1):
        new_entry = {
            "type": block.type,
            "format": block.format,
            "offset": self.entries[-1]["offset"] + self.entries[-1]["size"],
            "size": block.size,
            "creation_date": block.creation_date,
            "last_modification_date": block.last_modification_date,
            "last_access_date": datetime.now(),
            "comment": block.comment,
        }

    def __del__(self):
        self.handler.close()
