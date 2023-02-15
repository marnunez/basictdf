from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest import TestCase

import numpy as np

from basictdf.basictdf import Tdf, TdfEntry
from basictdf.tdfBlock import BlockType
from basictdf.tdfEvents import (
    Event,
    EventsDataType,
    TemporalEventsData,
    TemporalEventsDataFormat,
)
from basictdf.tdfTypes import BTSDate, BTSString, i32, u32
from basictdf.tdfUtils import OutsideOfContextError
from tests import test_file_feeder


class TestEntry(TestCase):
    def test_creation(self) -> None:
        date = datetime(2020, 1, 1, 1, 1)
        a = TdfEntry(
            BlockType.temporalEventsData,
            1,
            16,
            45,
            date,
            date,
            date,
            "comment",
        )
        self.assertEqual(a.type, BlockType.temporalEventsData)
        self.assertEqual(a.format, 1)
        self.assertEqual(a.offset, 16)
        self.assertEqual(a.size, 45)
        self.assertEqual(a.creation_date, date)
        self.assertEqual(a.last_modification_date, date)
        self.assertEqual(a.last_access_date, date)
        self.assertEqual(a.comment, "comment")

    def test_build(self) -> None:
        date = datetime(2020, 1, 1, 1, 1)
        b = BytesIO()
        # type
        u32.bwrite(b, BlockType.temporalEventsData.value)
        # format
        u32.bwrite(b, 1)
        # offset
        i32.bwrite(b, 16)
        # size
        i32.bwrite(b, 45)
        # creation_date
        BTSDate.bwrite(b, date)
        # last_modification_date
        BTSDate.bwrite(b, date)
        # last_access_date
        BTSDate.bwrite(b, date)
        # padd
        i32.bpad(b)
        # comment
        BTSString.bwrite(b, 256, "comment")
        b.seek(0, 0)
        a = TdfEntry._build(b)

        self.assertEqual(a.type, BlockType.temporalEventsData)
        self.assertEqual(a.format, 1)
        self.assertEqual(a.offset, 16)
        self.assertEqual(a.size, 45)
        self.assertEqual(a.creation_date, date)
        self.assertEqual(a.last_modification_date, date)
        self.assertEqual(a.last_access_date, date)
        self.assertEqual(a.comment, "comment")
        self.assertEqual(a.nBytes, 288)
        self.assertEqual(a.nBytes, len(b.getvalue()))

        c = BytesIO()
        a._write(c)
        self.assertEqual(b.getvalue(), c.getvalue())


class TestTdf(TestCase):
    def tearDown(self) -> None:
        for tdf in Path("tests/").glob("*.tdf"):
            tdf.unlink()

    def test_creation(self):
        with Tdf.new("tests/test.tdf").allow_write() as tdf:
            self.assertEqual(tdf.filePath.name, "test.tdf")
            self.assertEqual(len(tdf.entries), 14)
            for entry in tdf.entries:
                self.assertEqual(entry.type, BlockType.unusedSlot)
                self.assertEqual(entry.format, 0)
                self.assertEqual(entry.offset, 64 + 288 * len(tdf.entries))
                self.assertEqual(entry.size, 0)
                self.assertIsInstance(entry.creation_date, datetime)
                self.assertIsInstance(entry.last_modification_date, datetime)
                self.assertIsInstance(entry.last_access_date, datetime)
                self.assertEqual(entry.comment, "Generated by basicTDF")

        with tdf.filePath.open("rb") as f:
            content = f.read()

        self.assertEqual(len(content), 64 + 288 * len(tdf.entries))

    def test_add_block(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)

        tdf_file = Tdf.new("tests/add_test.tdf")

        with tdf_file.allow_write() as tdf:
            tdf.add_block(eventBlock, "My favourite event block")

        with tdf_file as tdf:
            self.assertEqual(len(tdf.entries), 14)
            self.assertEqual(tdf.entries[0].type, BlockType.temporalEventsData)
            self.assertEqual(
                tdf.entries[0].format, TemporalEventsDataFormat.standard.value
            )
            self.assertEqual(tdf.entries[0].comment, "My favourite event block")
            self.assertEqual(tdf.entries[0].offset, 64 + 288 * len(tdf.entries))
            self.assertEqual(tdf.entries[0].size, eventBlock.nBytes)
            self.assertIsInstance(tdf.entries[0].creation_date, datetime)
            self.assertIsInstance(tdf.entries[0].last_modification_date, datetime)
            self.assertIsInstance(tdf.entries[0].last_access_date, datetime)
            for entry in tdf.entries[1:]:
                self.assertEqual(entry.type, BlockType.unusedSlot)
                self.assertEqual(
                    entry.offset, 64 + 288 * len(tdf.entries) + tdf.entries[0].size
                )

            block = tdf.get_block(BlockType.temporalEventsData)

            for before, after in zip(eventBlock.events, block.events):
                self.assertEqual(before.type, after.type)
                np.testing.assert_equal(before.values, after.values)

    def test_remove_block(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)

        tdf_file = Tdf.new("tests/remove_test.tdf")

        with tdf_file.allow_write() as tdf:
            tdf.add_block(eventBlock, "My favourite event block")
            oldSize = tdf.nBytes

        # Can't remove a block in read-only mode
        with self.assertRaises(ValueError):
            with tdf_file as tdf:
                tdf.remove_block(BlockType.temporalEventsData)

        with tdf_file.allow_write() as tdf:
            tdf.remove_block(BlockType.temporalEventsData)
            self.assertEqual(len(tdf.entries), 14)
            for entry in tdf.entries:
                self.assertEqual(entry.type, BlockType.unusedSlot)
                self.assertEqual(entry.format, 0)
                self.assertEqual(entry.offset, 64 + 288 * len(tdf.entries))
                self.assertEqual(entry.size, 0)
                self.assertIsInstance(entry.creation_date, datetime)
                self.assertIsInstance(entry.last_modification_date, datetime)
                self.assertIsInstance(entry.last_access_date, datetime)
                self.assertEqual(entry.comment, "Generated by basicTDF")
            self.assertEqual(tdf.nBytes, oldSize - eventBlock.nBytes)

    def test_replace_block(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)

        tdf_file = Tdf.new("tests/test.tdf")

        with tdf_file.allow_write() as tdf:
            tdf.add_block(eventBlock, "My favourite event block")

        event.values = [4, 5, 6]
        self.assertEqual(event.values, eventBlock.events[0].values)

        with tdf_file.allow_write() as tdf:
            oldSize = tdf.nBytes
            tdf.replace_block(eventBlock, "My modified favourite event block")

        with tdf_file as tdf:
            self.assertEqual(tdf.nBytes, oldSize)
            self.assertEqual(len(tdf.entries), 14)
            self.assertEqual(tdf.entries[0].type, BlockType.temporalEventsData)
            self.assertEqual(
                tdf.entries[0].format, TemporalEventsDataFormat.standard.value
            )
            self.assertEqual(
                tdf.entries[0].comment, "My modified favourite event block"
            )
            self.assertEqual(tdf.entries[0].offset, 64 + 288 * len(tdf.entries))
            self.assertEqual(tdf.entries[0].size, eventBlock.nBytes)
            event = tdf.events[0]
            np.testing.assert_equal(event.values, [4, 5, 6])

    def test_replace_using_setters(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)

        tdf_file = Tdf.new("tests/replace_test.tdf")

        with tdf_file.allow_write() as tdf:
            tdf.add_block(eventBlock, "My favourite event block")

        event.values = [4, 5, 6]
        self.assertEqual(event.values, eventBlock.events[0].values)

        with tdf_file.allow_write() as tdf:
            oldSize = tdf.nBytes
            tdf.events = eventBlock

        with tdf_file as tdf:
            self.assertEqual(tdf.nBytes, oldSize)
            self.assertEqual(len(tdf.entries), 14)
            self.assertEqual(tdf.entries[0].type, BlockType.temporalEventsData)
            self.assertEqual(
                tdf.entries[0].format, TemporalEventsDataFormat.standard.value
            )

            # when using setters, the comment from the original block is kept
            self.assertEqual(tdf.entries[0].comment, "My favourite event block")
            self.assertEqual(tdf.entries[0].offset, 64 + 288 * len(tdf.entries))
            self.assertEqual(tdf.entries[0].size, eventBlock.nBytes)
            event = tdf.events[0]
            np.testing.assert_equal(event.values, [4, 5, 6])

    def test_cannot_write_outside_context(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)

        tdf_file = Tdf.new("tests/replace_test.tdf")

        with self.assertRaises(OutsideOfContextError):
            tdf_file.add_block(eventBlock, "My favourite event block")

        with self.assertRaises(OutsideOfContextError):
            tdf_file.events = eventBlock

    def test_read_only_property(self) -> None:
        event = Event("jaja", values=[1, 2, 3], type=EventsDataType.eventSequence)
        eventBlock = TemporalEventsData()
        eventBlock.events.append(event)


class TestRealTdf(TestCase):
    def test_read(self) -> None:
        for file, metadata in test_file_feeder():
            tdf_file = Tdf(file)
            with tdf_file as tdf:
                self.assertEqual(tdf.nBytes, file.stat().st_size)
                self.assertEqual(len(tdf.entries), 14)
                self.assertEqual(len(tdf), 8)