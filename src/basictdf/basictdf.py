from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional, Type, Union

from basictdf.tdfBlock import (
    AnalogData,
    AnthropometricData,
    Block,
    BlockType,
    CalibrationData2D,
    ForcePlatformsCalibrationData2D,
    GeneralCalibrationData,
    NotDefinedBlock,
    UnusedBlock,
    VolumetricData,
)
from basictdf.tdfForcePlatformsData import ForcePlatformsDataBlock
from basictdf.tdfCalibrationData import CalibrationDataBlock
from basictdf.tdfData2D import Data2D
from basictdf.tdfData3D import Data3D
from basictdf.tdfEMG import EMG
from basictdf.tdfEvents import TemporalEventsData
from basictdf.tdfForce3D import ForceTorque3D
from basictdf.tdfOpticalSystem import OpticalSetupBlock
from basictdf.tdfTypes import BTSDate, BTSString, i32, u32
from basictdf.tdfUtils import (
    provide_context_if_needed,
    raise_if_outside_context,
    raise_if_outside_write_context,
)
from basictdf.tdfForcePlatformsCalibration import ForcePlatformsCalibrationDataBlock


def _get_block_class(block_type: BlockType) -> Type[Block]:
    if block_type == BlockType.unusedSlot:
        return UnusedBlock
    elif block_type == BlockType.notDefined:
        return NotDefinedBlock
    elif block_type == BlockType.calibrationData:
        return CalibrationDataBlock
    elif block_type == BlockType.calibrationData2D:
        return CalibrationData2D
    elif block_type == BlockType.data2D:
        return Data2D
    elif block_type == BlockType.data3D:
        return Data3D
    elif block_type == BlockType.opticalSystemConfiguration:
        return OpticalSetupBlock
    elif block_type == BlockType.forcePlatformsCalibrationData:
        return ForcePlatformsCalibrationDataBlock
    elif block_type == BlockType.forcePlatformsCalibrationData2D:
        return ForcePlatformsCalibrationData2D
    elif block_type == BlockType.forcePlatformsData:
        return ForcePlatformsDataBlock
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
    """A jumptable type entry for a data block."""

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
    ) -> None:
        self.type = type
        self.format = format
        self.offset = offset
        self.size = size
        self.creation_date = creation_date
        self.last_modification_date = last_modification_date
        self.last_access_date = last_access_date
        self.comment = comment
        self.nBytes = 8 * 4 + 256

    def _write(self, file) -> None:
        u32.bwrite(file, self.type.value)
        u32.bwrite(file, self.format)
        i32.bwrite(file, self.offset)
        i32.bwrite(file, self.size)
        BTSDate.bwrite(file, self.creation_date)
        BTSDate.bwrite(file, self.last_modification_date)
        BTSDate.bwrite(file, self.last_access_date)
        i32.bpad(file)
        BTSString.bwrite(file, 256, self.comment)

    @staticmethod
    def _build(file) -> "TdfEntry":
        type_ = BlockType(u32.bread(file))
        format = u32.bread(file)
        offset = i32.bread(file)
        size = i32.bread(file)
        creation_date = BTSDate.bread(file)
        last_modification_date = BTSDate.bread(file)
        last_access_date = BTSDate.bread(file)
        i32.skip(file)
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

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TdfEntry):
            return False
        return (
            self.type == o.type
            and self.format == o.format
            and self.offset == o.offset
            and self.size == o.size
            and self.creation_date == o.creation_date
            and self.last_modification_date == o.last_modification_date
            and self.last_access_date == o.last_access_date
            and self.comment == o.comment
        )

    def __repr__(self) -> str:
        return (
            f"TdfEntry(type={self.type}, format={self.format}, offset={self.offset}, "
            f"size={self.size}, creation_date={self.creation_date}, "
            f"last_modification_date={self.last_modification_date}, "
            f"last_access_date={self.last_access_date}, comment={self.comment})"
        )


class Tdf:
    SIGNATURE = b"\x82K`A\xd3\x11\x84\xca`\x00\xb6\xac\x16h\x0c\x08"

    def __init__(self, filename: Union[Path, str]) -> None:
        self.file_path = Path(filename)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} not found")

        self._mode = "rb"
        self._inside_context = False

    def allow_write(self) -> "Tdf":
        """Allow writing to the file."""
        self._mode = "r+b"
        return self

    def __enter__(self) -> "Tdf":
        self._inside_context = True
        self.handler: IO[bytes] = self.file_path.open(self._mode)

        self.signature = self.handler.read(len(self.SIGNATURE))

        if self.signature != self.SIGNATURE:
            raise Exception("Invalid TDF file")

        self.version = u32.bread(self.handler)
        self.nEntries = i32.bread(self.handler)

        # pad 8 bytes
        i32.skip(self.handler, 2)

        self.creation_date = BTSDate.bread(self.handler)
        self.last_modification_date = BTSDate.bread(self.handler)
        self.last_access_date = BTSDate.bread(self.handler)

        # pad 20 bytes
        i32.skip(self.handler, 5)

        self.entries = [TdfEntry._build(self.handler) for _ in range(self.nEntries)]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._inside_context = False
        self._mode = "rb"
        self.handler.close()

    @property
    @provide_context_if_needed
    def blocks(self) -> List[Block]:
        """Get all blocks in the file."""
        return [self.get_block(entry.type) for entry in self.entries]

    @provide_context_if_needed
    def get_block(self, index_or_type: Union[BlockType, int]) -> Optional[Type[Block]]:
        """Get a block from the TDF file."""

        if isinstance(index_or_type, int):
            if 0 <= index_or_type < len(self.entries):
                entry = self.entries[index_or_type]
            else:
                raise IndexError(f"Index {index_or_type} out of range")

        elif isinstance(index_or_type, BlockType):
            entry = next((e for e in self.entries if e.type == index_or_type), None)
            if entry is None:
                raise Exception(f"Block {index_or_type} not found")

        else:
            raise TypeError(f"Expected int or BlockType, got {type(index_or_type)}")

        self.handler.seek(entry.offset, 0)
        block_class = _get_block_class(entry.type)
        return block_class._build(self.handler, entry.format)

    def __getitem__(
        self, index_or_type: Union[BlockType, int]
    ) -> Optional[Type[Block]]:
        return self.get_block(index_or_type)

    @property
    @provide_context_if_needed
    def data3D(self) -> Optional[Data3D]:
        """
        Convenience property to get/set the 3D data block.
        """
        return self.get_block(BlockType.data3D)

    @data3D.setter
    @raise_if_outside_context
    def data3D(self, data: Data3D) -> None:
        self.replace_block(data) if self.has_data3D else self.add_block(data)

    @property
    @provide_context_if_needed
    def has_data3D(self) -> bool:
        """Check if the file has a 3D data block."""
        return any(entry.type == BlockType.data3D for entry in self.entries)

    @property
    @provide_context_if_needed
    def force_and_torque(self) -> Optional[ForceTorque3D]:
        """Convenience property to get/set the force and torque data block."""
        return self.get_block(BlockType.forceAndTorqueData)

    @force_and_torque.setter
    @raise_if_outside_write_context
    def force_and_torque(self, data: ForceTorque3D) -> None:
        self.replace_block(data) if self.has_force_and_torque else self.add_block(data)

    @property
    @provide_context_if_needed
    def has_force_and_torque(self) -> bool:
        """Check if the file has a force and torque data block."""
        return any(entry.type == BlockType.forceAndTorqueData for entry in self.entries)

    @property
    @provide_context_if_needed
    def events(self) -> Optional[TemporalEventsData]:
        """Convenience property to get/set the events data block."""
        return self.get_block(BlockType.temporalEventsData)

    @events.setter
    @raise_if_outside_write_context
    def events(self, data: TemporalEventsData) -> None:
        self.replace_block(data) if self.has_events else self.add_block(data)

    @property
    @provide_context_if_needed
    def has_events(self) -> bool:
        """Check if the TDF file has an events block"""
        return any(i for i in self.entries if i.type == BlockType.temporalEventsData)

    @property
    @provide_context_if_needed
    def emg(self) -> Optional[EMG]:
        """Convenience property to get/set the EMG data block."""
        return self.get_block(EMG.type)

    @emg.setter
    @raise_if_outside_write_context
    def emg(self, data: EMG) -> None:
        self.replace_block(data) if self.has_emg else self.add_block(data)

    @property
    @provide_context_if_needed
    def has_emg(self) -> bool:
        """Check if the TDF file has an EMG block"""
        return any(
            entry.type == BlockType.electromyographicData for entry in self.entries
        )

    @property
    @provide_context_if_needed
    def calibrationData(self) -> Optional[CalibrationDataBlock]:
        """Convenience property to get/set the calibration data block."""
        return self.get_block(CalibrationDataBlock.type)

    @raise_if_outside_write_context
    def add_block(
        self, newBlock: Block, comment: str = "Generated by basicTDF"
    ) -> None:
        """Adds a block to the TDF file

        Args:
            newBlock (Block): the block to be added.
            comment (str, optional): A description for the block entry.
            Defaults to "Generated by basicTDF".

        Raises:
            PermissionError: the TDF is read only
            ValueError: there's already a block of the same type
            ValueError: block limit reached (14 as per BTS's implementation)
            IOError: unused empty blocks in the middle of the file
        """
        if self._mode == "rb":
            raise PermissionError(
                "Can't add blocks, this file was opened in read-only mode"
            )

        try:
            if self.get_block(newBlock.type):
                raise ValueError(
                    (
                        f"There's already a block of this type {newBlock.type}"
                        " .Remove it first"
                    )
                )
        except Exception:
            pass

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

        # ensure the file is the correct size
        # and that the changes are written to disk
        self.handler.flush()

    @raise_if_outside_write_context
    def remove_block(self, type: Block) -> None:
        """Remove a block of the given type from the file. Removing a block
        implies:

        - Removing the entry
        - Updating all subsequent unused slots's offset
          (subtracting the size of the removed block)
        - Inserting a new unused slot entry at the end
          (with the previous slot offset + size as offset)
        - If there is info after the block, move it
          block_to_remove.size up

        """
        if "+" not in self._mode:
            raise PermissionError(
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
        self.handler.flush()

    @staticmethod
    def new(filename: str) -> "Tdf":
        """Creates a new TDF file"""
        filePath = Path(filename)
        if filePath.exists():
            raise FileExistsError("File already exists")

        nEntries = 14
        date = datetime.now()
        with filePath.open("wb") as f:
            # signature
            f.write(Tdf.SIGNATURE)
            # version
            i32.bwrite(f, 1)
            # nEntries
            i32.bwrite(f, nEntries)
            # reserved
            i32.bpad(f, 2)
            # creation date
            BTSDate.bwrite(f, date)
            # last modification date
            BTSDate.bwrite(f, date)
            # last access date
            BTSDate.bwrite(f, date)
            # reserved
            i32.bpad(f, 5)

            # start entries
            entryOffset = 64

            # all entries start with offset to the where entries stop
            blockOffset = entryOffset + nEntries * 288

            for _ in range(nEntries):
                # type
                u32.bwrite(f, 0)
                # format
                u32.bwrite(f, 0)
                # offset
                i32.bwrite(f, blockOffset)
                # size
                i32.bwrite(f, 0)
                # creation date
                BTSDate.bwrite(f, date)
                # last modification date
                BTSDate.bwrite(f, date)
                # last access date
                BTSDate.bwrite(f, date)
                # reserved
                i32.bpad(f, 1)
                # comment
                BTSString.bwrite(f, 256, "Generated by basicTDF")

        return Tdf(filePath)

    @raise_if_outside_write_context
    def replace_block(self, newBlock: Block, comment: Optional[str] = None) -> None:
        """Replace a block of the same type with a new one. This is done by
        removing the old block and adding the new one."""

        old_entry = next((i for i in self.entries if i.type == newBlock.type), None)

        if old_entry is None:
            raise ValueError(f"No block of type {newBlock.type} found")

        comment = comment if comment is not None else old_entry.comment

        self.remove_block(newBlock.type)
        self.add_block(newBlock, comment)

    @property
    def nBytes(self) -> int:
        """Return the size of the TDF file in bytes"""
        return self.file_path.stat().st_size

    def __len__(self) -> int:
        """Return the number of blocks in the TDF file
        that are not of type unusedSlot
        """
        return sum(1 for i in self.entries if i.type != BlockType.unusedSlot)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Tdf):
            return False
        return (
            self.version == o.version
            and self.nEntries == o.nEntries
            and self.blocks == o.blocks
        )

    @provide_context_if_needed
    def __repr__(self) -> str:
        return f"Tdf({self.file_path}, nEntries={self.nEntries}, nBytes={self.nBytes}"
