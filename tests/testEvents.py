from unittest import TestCase
from basictdf.tdfEvents import (
    Event,
    EventsDataType,
    TemporalEventsData,
    TemporalEventsDataFormat,
)
from io import BytesIO
import numpy as np


class TestEvent(TestCase):
    def test_creation(self):
        a = Event("hola")
        self.assertEqual(a.label, "hola")
        np.testing.assert_equal(a.values, np.array([], dtype="<f4"))
        with self.assertRaises(TypeError):
            b = Event("hola", values=1)

        with self.assertRaises(TypeError):
            b = Event("hola", values=[1, 2, 3], type=EventsDataType.singleEvent)

    def test_build(self):
        b = b"hola" + b"\x00" * 252
        b += b"\x01\x00\x00\x00"  # type
        b += b"\x02\x00\x00\x00"  # nValues
        b += np.array([3, 4], dtype="<f4").tobytes()
        e = Event.build(BytesIO(b))
        self.assertEqual(e.label, "hola")
        self.assertEqual(e.type, EventsDataType.eventSequence)
        self.assertEqual(len(e.values), 2)
        np.testing.assert_equal(e.values, np.array([3.0, 4.0]))
        self.assertEqual(e.nBytes, len(b))

    def test_write(self):
        e = Event("hola", values=[3, 4], type=EventsDataType.eventSequence)
        a = BytesIO()
        e.write(a)

        b = b"hola" + b"\x00" * 252
        b += b"\x01\x00\x00\x00"  # type
        b += b"\x02\x00\x00\x00"  # nValues
        b += np.array([3, 4], dtype="<f4").tobytes()
        self.assertEqual(a.getvalue(), b)
        self.assertEqual(e.nBytes, len(b))


class TestTemporalEventsDataBlock(TestCase):
    def test_creation(self):
        a = TemporalEventsData()
        self.assertEqual(a.format, TemporalEventsDataFormat.standard)
        self.assertEqual(a.start_time, 0)
        self.assertEqual(a.nBytes, 8)
        self.assertEqual(len(a), 0)
        ev = Event("hola", values=[3, 4], type=EventsDataType.eventSequence)

        a.events.append(ev)
        self.assertEqual(a.nBytes, 8 + ev.nBytes)
        self.assertEqual(a.events[0], ev)
        self.assertEqual(len(a), 1)

    def test_build(self):
        b = BytesIO()
        b.write(b"\x02\x00\x00\x00")  # nEvents
        b.write(b"\x00\x00\x00\x00")  # start_time
        Event("hola", values=[3, 4], type=EventsDataType.eventSequence).write(b)
        Event("hola2", values=[5, 6], type=EventsDataType.eventSequence).write(b)
        b.seek(0, 0)
        d = TemporalEventsData.build(b, format=TemporalEventsDataFormat.standard)
        self.assertEqual(d.format, TemporalEventsDataFormat.standard)
        self.assertEqual(d.start_time, 0.0)
        self.assertEqual(len(d), 2)
        self.assertEqual(d.nBytes, b.tell())
        self.assertEqual(d.nBytes, 8 + d.events[0].nBytes + d.events[1].nBytes)

    def test_write(self):
        b = BytesIO()
        b.write(b"\x02\x00\x00\x00")  # nEvents
        b.write(b"\x00\x00\x00\x00")  # start_time
        Event("hola", values=[3, 4], type=EventsDataType.eventSequence).write(b)
        Event("hola2", values=[5, 6], type=EventsDataType.eventSequence).write(b)

        c = TemporalEventsData(format=TemporalEventsDataFormat.standard, start_time=0.0)
        c.events = [
            Event("hola", values=[3, 4], type=EventsDataType.eventSequence),
            Event("hola2", values=[5, 6], type=EventsDataType.eventSequence),
        ]
        d = BytesIO()
        c.write(d)

        self.assertEqual(d.getvalue(), b.getvalue())
        self.assertEqual(len(d.getvalue()), c.nBytes)
