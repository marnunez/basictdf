from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from basictdf.tdfBlock import (
    AnalogData,
    AnthropometricData,
    Block,
    BlockType,
    CalibrationData,
    CalibrationData2D,
    Data2D,
    ForcePlatformsCalibrationData,
    ForcePlatformsCalibrationData2D,
    ForcePlatformsData,
    GeneralCalibrationData,
    OpticalSystemConfiguration,
    VolumetricData,
)
from basictdf.tdfEMG import EMG
from basictdf.tdfData3D import Data3D
from basictdf.tdfForce3D import ForceTorque3D
from basictdf.tdfEvents import TemporalEventsData
from basictdf.tdfForce3D import ForceTorque3D
from basictdf.tdfTypes import BTSDate, BTSString, Int32, Uint32


def get_block_class(block_type):
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
        return EMG
    elif block_type == BlockType.forceAndTorqueData:
        return ForceTorque3D
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


class TdfEntry:
    def __init__(
        self,
        type: BlockType,
        format: int,
        offset: int,
        size: int,
        creation_date: datetime,
        last_modification_date: datetime,
        last_access_date: datetime,
        comment: str,
    ):
        self.type = type
        self.format = format
        self.offset = offset
        self.size = size
        self.creation_date = creation_date
        self.last_modification_date = last_modification_date
        self.last_access_date = last_access_date
        self.comment = comment
        self.nBytes = 8 * 4 + 256

    def _write(self, file):
        Uint32.bwrite(file, self.type.value)
        Uint32.bwrite(file, self.format)
        Int32.bwrite(file, self.offset)
        Int32.bwrite(file, self.size)
        BTSDate.bwrite(file, self.creation_date)
        BTSDate.bwrite(file, self.last_modification_date)
        BTSDate.bwrite(file, self.last_access_date)
        Int32.bpad(file)
        BTSString.bwrite(file, 256, self.comment)

    @staticmethod
    def _build(file):
        type_ = BlockType(Uint32.bread(file))
        format = Uint32.bread(file)
        offset = Int32.bread(file)
        size = Int32.bread(file)
        creation_date = BTSDate.bread(file)
        last_modification_date = BTSDate.bread(file)
        last_access_date = BTSDate.bread(file)
        Int32.skip(file)
        comment = BTSString.bread(file, 256)
        return TdfEntry(
            type_,
            format,
            offset,
            size,
            creation_date,
            last_modification_date,
            last_access_date,
            comment,
        )


class Tdf:
    SIGNATURE = b"\x82K`A\xd3\x11\x84\xca`\x00\xb6\xac\x16h\x0c\x08"

    def __init__(self, filename, mode="rb"):
        self.filePath = Path(filename)

        if "b" not in mode:
            mode += "b"

        assert mode in ["rb", "r+b"], ValueError(
            f"Invalid mode {mode}. Must be 'rb' or 'r+b'"
        )

        self.mode = mode

        if not self.filePath.exists():
            raise FileNotFoundError(f"File {self.filePath} not found")

    def __enter__(self):
        self.handler = self.filePath.open(self.mode)

        self.signature = self.handler.read(len(self.SIGNATURE))

        if self.signature != self.SIGNATURE:
            raise Exception("Invalid TDF file")

        self.version = Uint32.bread(self.handler)
        self.nEntries = Int32.bread(self.handler)

        # pad 8 bytes
        Int32.skip(self.handler, 2)

        self.creation_date = BTSDate.bread(self.handler)
        self.last_modification_date = BTSDate.bread(self.handler)
        self.last_access_date = BTSDate.bread(self.handler)

        # pad 20 bytes
        Int32.skip(self.handler, 5)

        self.entries = [TdfEntry._build(self.handler) for _ in range(self.nEntries)]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.close()

    def get_block(self, type):
        for entry in self.entries:
            if entry.type == type:
                self.handler.seek(entry.offset, 0)
                block_class = get_block_class(entry.type)
                return block_class._build(self.handler, entry.format)
        return None

    @property
    def data3D(self):
        return self.get_block(BlockType.data3D)

    @property
    def has_data3D(self):
        with self:
            return any(i for i in self.entries if i.type == BlockType.data3D)

    @property
    def force_and_torque(self):
        return self.get_block(BlockType.forceAndTorqueData)

    @property
    def has_force_and_torque(self):
        with self:
            return any(
                i for i in self.entries if i.type == BlockType.forceAndTorqueData
            )

    @property
    def events(self):
        return self.get_block(BlockType.temporalEventsData)

    @property
    def has_events(self):
        with self:
            return any(
                i for i in self.entries if i.type == BlockType.temporalEventsData
            )

    @property
    def emg(self):
        return self.get_block(BlockType.electromyographicData)

    @property
    def has_emg(self):
        with self:
            return any(
                i for i in self.entries if i.type == BlockType.electromyographicData
            )

    def add_block(self, newBlock: Block, comment: str = "Generated by basicTDF"):
        if self.mode == "rb":
            raise ValueError("Can't add blocks, this file was opened in read-only mode")

        if self.get_block(newBlock.type):
            raise ValueError(
                f"There's already a block of this type {newBlock.type}. Remove it first"
            )

        # find first unused slot
        try:
            unusedBlockPos = next(
                n for n, i in enumerate(self.entries) if i.type == BlockType.unusedSlot
            )
        except StopIteration:
            raise ValueError(f"Block limit reached ({len(self.entries)})")

        # write new entry with the offset of that unused slot
        new_entry = TdfEntry(
            type=newBlock.type,
            format=newBlock.format.value,
            offset=self.entries[unusedBlockPos].offset,
            size=newBlock.nBytes,
            creation_date=newBlock.creation_date,
            last_modification_date=newBlock.last_modification_date,
            last_access_date=datetime.now(),
            comment=comment,
        )

        # replace the entry
        self.entries[unusedBlockPos] = new_entry

        # write new entry
        self.handler.seek(64 + 288 * unusedBlockPos, 0)
        new_entry._write(self.handler)

        # update all unused slots's offset
        for n, entry in enumerate(
            self.entries[unusedBlockPos + 1 :], start=unusedBlockPos + 1
        ):
            if entry.type == BlockType.unusedSlot:
                entry.offset = new_entry.offset + new_entry.size
                self.handler.seek(64 + 288 * n, 0)
                entry._write(self.handler)
            else:
                raise IOError("All unused slots must be at the end of the file")

        # write new block
        self.handler.seek(new_entry.offset, 0)
        newBlock._write(self.handler)
        self.handler.flush()

    def remove_block(self, type: Block):
        """Remove a block of the given type from the file. Removing a block implies:

        - Removing the entry
        - Updating all subsequent unused slots's offset (subtracting the size of the removed block)
        - Inserting a new unused slot entry at the end (with the previous slot offset + size as offset)
        - If there is info after the block, move it block_to_remove.size up

        """

        if self.mode == "rb":
            raise ValueError(
                "Can't remove blocks, this file was opened in read-only mode"
            )

        # find block
        try:
            oldEntryPos, oldEntry = next(
                (n, i) for n, i in enumerate(self.entries) if i.type == type
            )
        except StopIteration:
            raise ValueError(f"No block of type {type} found")

        # calculate new offset for the next unused slot
        newOffset = (
            self.entries[-1].offset if oldEntryPos != 0 else (64 + 288 * self.nEntries)
        )

        # delete entry
        self.entries.remove(oldEntry)
        self.handler.seek(64 + 288 * oldEntryPos, 0)
        # update all the offsets of the entries preceding the removed one
        for entry in self.entries[oldEntryPos:]:
            entry.offset -= oldEntry.size
            entry._write(self.handler)

        # add new unused slot at the end
        date = datetime.now()
        newEntry = TdfEntry(
            type=BlockType.unusedSlot,
            format=0,
            offset=newOffset,
            size=0,
            creation_date=date,
            last_modification_date=date,
            last_access_date=date,
            comment="Generated by basicTDF",
        )
        self.entries.append(newEntry)
        newEntry._write(self.handler)

        self.handler.seek(oldEntry.offset + oldEntry.size, 0)
        temp = self.handler.read()
        self.handler.seek(oldEntry.offset, 0)
        self.handler.write(temp)
        self.handler.truncate()

    @staticmethod
    def new(filename: str):
        filePath = Path(filename)
        if filePath.exists():
            raise FileExistsError("File already exists")

        nEntries = 14
        date = datetime.now()
        with filePath.open("wb") as f:
            # signature
            f.write(Tdf.SIGNATURE)
            # version
            Int32.bwrite(f, 1)
            # nEntries
            Int32.bwrite(f, nEntries)
            # reserved
            Int32.bpad(f, 2)
            # creation date
            BTSDate.bwrite(f, date)
            # last modification date
            BTSDate.bwrite(f, date)
            # last access date
            BTSDate.bwrite(f, date)
            # reserved
            Int32.bpad(f, 5)

            # start entries
            entryOffset = 64

            # all entries start with offset to the where entries stop
            blockOffset = entryOffset + nEntries * 288

            for _ in range(nEntries):
                # type
                Uint32.bwrite(f, 0)
                # format
                Uint32.bwrite(f, 0)
                # offset
                Int32.bwrite(f, blockOffset)
                # size
                Int32.bwrite(f, 0)
                # creation date
                BTSDate.bwrite(f, date)
                # last modification date
                BTSDate.bwrite(f, date)
                # last access date
                BTSDate.bwrite(f, date)
                # reserved
                Int32.bpad(f, 1)
                # comment
                BTSString.bwrite(f, 256, "Generated by basicTDF")

        return Tdf(filePath, mode="r+b")

    @property
    def nBytes(self):
        return self.filePath.stat().st_size

    def __repr__(self):
        return f"Tdf({self.filePath}),{self.nEntries} entries"
