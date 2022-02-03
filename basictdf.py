import mmap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tdfBlock import (
    AnalogData,
    AnthropometricData,
    BlockType,
    CalibrationData,
    CalibrationData2D,
    Data2D,
    ElectromyographicData,
    ForceAndTorqueData,
    ForcePlatformsCalibrationData,
    ForcePlatformsCalibrationData2D,
    ForcePlatformsData,
    GeneralCalibrationData,
    OpticalSystemConfiguration,
    VolumetricData,
)
from tdfData3D import Data3D
from tdfEvents import TemporalEventsData
from tdfTypes import BTSDate, BTSString, Int32, Uint32


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


@dataclass
class TdfEntry:
    type: BlockType
    format: int
    offset: int
    size: int
    creation_date: datetime
    last_modification_date: datetime
    last_access_date: datetime
    comment: str
    nBytes = 8 * 4 + 256

    def write(self, file):
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
    def build(file):
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
        self.filename = Path(filename)

        if "b" not in mode:
            mode += "b"

        assert mode in ["rb", "r+b"], ValueError(
            f"Invalid mode {mode}. Must be 'rb' or 'r+b'"
        )

        self.mode = mode

        if not self.filename.exists():
            raise FileNotFoundError(f"File {self.filename} not found")

    def __enter__(self):
        self.filehandler = self.filename.open(self.mode)
        self.handler = mmap.mmap(
            self.filehandler.fileno(),
            0,
            access=mmap.ACCESS_WRITE if self.mode == "r+b" else mmap.ACCESS_READ,
        )
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

        self.entries = [TdfEntry.build(self.handler) for _ in range(self.nEntries)]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.handler.flush()
        self.handler.close()
        self.filehandler.close()

    def get_block(self, type):
        for entry in self.entries:
            if entry.type == type:
                self.handler.seek(entry.offset, 0)
                block_class = get_block_class(entry.type)
                return block_class.build(self.handler, entry.format)
        return None

    @property
    def data3D(self):
        return self.get_block(BlockType.data3D)

    @property
    def events(self):
        return self.get_block(BlockType.temporalEventsData)

    def add_block(self, newBlock, comment="Generated by basicTDF"):
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

        # if it's the first entry, the offset is 64
        lastBlockOffset = (
            self.entries[unusedBlockPos - 1].offset if unusedBlockPos > 0 else 64
        )

        # if it's the first entry, the size is the length of all entries
        lastBlockSize = (
            self.entries[unusedBlockPos - 1].size
            if unusedBlockPos > 0
            else 288 * len(self.entries)
        )

        new_entry = TdfEntry(
            type=newBlock.type,
            format=newBlock.format.value,
            offset=lastBlockOffset + lastBlockSize,
            size=newBlock.nBytes,
            creation_date=newBlock.creation_date,
            last_modification_date=newBlock.last_modification_date,
            last_access_date=datetime.now(),
            comment=comment,
        )

        self.entries[unusedBlockPos] = new_entry

        # write new entry
        self.handler.seek(64 + 288 * unusedBlockPos, 0)
        new_entry.write(self.handler)

        # flush updated entry
        self.handler.flush(64 + 288 * unusedBlockPos, 288)

        # write new block
        self.handler.resize(self.handler.size() + newBlock.nBytes)
        self.handler.seek(new_entry.offset, 0)
        newBlock.write(self.handler)

        # flush block
        self.handler.flush(new_entry.offset, new_entry.size)

    def remove_block(self, type):
        """Remove a block of the given type"""

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

        # overwrite entry with unused slot
        date = datetime.now()
        new_entry = TdfEntry(
            type=BlockType.unusedSlot,
            format=0,
            offset=64 + 288 * len(self.entries),
            size=0,
            creation_date=date,
            last_modification_date=date,
            last_access_date=date,
            comment="Generated by basicTDF",
        )

        # update entry
        self.entries[oldEntryPos] = new_entry
        self.handler.seek(64 + 288 * oldEntryPos, 0)
        new_entry.write(self.handler)

        # flush updated entry
        self.handler.flush(64 + 288 * oldEntryPos, 288)

        # update all the offsets of the entries preceding the removed one, unless they're unused
        for entry in self.entries[oldEntryPos + 1 :]:
            if entry.type == BlockType.unusedSlot:
                continue
            entry.offset -= oldEntry.size
            entry.write(self.handler)

        # flush updated entries
        self.handler.flush(
            64 + 288 * (oldEntryPos + 1), 288 * (len(self.entries) - (oldEntryPos + 1))
        )

        # move everything after the removed block up by oldBlock.size
        self.handler[oldEntry.offset : -oldEntry.size] = self.handler[
            oldEntry.offset + oldEntry.size :
        ]
        self.handler.resize(self.handler.size() - oldEntry.size)

        # if there's data to update, flush it
        if self.handler.size() - oldEntry.offset:
            self.handler.flush(oldEntry.offset, self.handler.size() - oldEntry.offset)

    @staticmethod
    def new(filename):
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

        return Tdf(filename, mode="r+b")

    @property
    def nBytes(self):
        return self.handler.size()

    def __repr__(self):
        return f"Tdf({self.filename}),{self.nEntries} entries"
