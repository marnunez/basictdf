__doc__ = "Events data module."

from enum import Enum
from basictdf.tdfBlock import Block
from basictdf.tdfTypes import BTSString, Uint32, Int32, Float32
from basictdf.tdfUtils import is_iterable
from basictdf.tdfBlock import BlockType
import numpy as np


class TemporalEventsDataFormat(Enum):
    unknown = 0
    standard = 1


class EventsDataType(Enum):
    singleEvent = 0
    eventSequence = 1


class Event:
    def __init__(self, label, values=[], type=EventsDataType.singleEvent):
        self.label = label
        self.type = type
        if not is_iterable(values):
            raise TypeError("Values must be iterable")
        if isinstance(values, np.ndarray) and values.dtype == np.dtype("<f4"):
            self.values = values
        else:
            self.values = np.array(values, dtype="<f4")

        if len(values) > 1 and type == EventsDataType.singleEvent:
            raise TypeError("Values must be a single value for singleEvent")

    def _write(self, stream):
        BTSString.bwrite(stream, 256, self.label)
        Uint32.bwrite(stream, self.type.value)  # type
        Uint32.bwrite(stream, len(self.values))  # nItems
        Float32.bwrite(stream, self.values)

    @staticmethod
    def _build(stream):
        label = BTSString.bread(stream, 256)
        type_ = EventsDataType(Uint32.bread(stream))
        nItems = Int32.bread(stream)
        values = np.array([Float32.bread(stream) for _ in range(nItems)])
        return Event(label, values, type_)

    def __len__(self):
        return len(self.values)

    @property
    def nBytes(self):
        return 256 + 4 + 4 + len(self.values) * 4

    def __repr__(self):
        return f"Event(label={self.label}, type={self.type}, items={self.values})"


class TemporalEventsData(Block):
    def __init__(self, format=TemporalEventsDataFormat.standard, start_time=0.0):
        super().__init__(BlockType.temporalEventsData)
        self.format = format
        self.start_time = start_time
        self.events = []

    @staticmethod
    def _build(stream, format):

        format = TemporalEventsDataFormat(format)
        nEvents = Int32.bread(stream)
        start_time = Float32.bread(stream)

        t = TemporalEventsData(format, start_time)
        t.events = [Event._build(stream) for _ in range(nEvents)]

        return t

    def _write(self, stream):
        Int32.bwrite(stream, len(self.events))
        Float32.bwrite(stream, self.start_time)
        for event in self.events:
            event._write(stream)

    @property
    def nBytes(self):
        return 4 + 4 + sum(i.nBytes for i in self.events)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return self.events[item]

    def __iter__(self):
        return iter(self.events)

    def __repr__(self):
        return f"TemporalEventsData(format={self.format}, nEvents={len(self.events)}, start_time={self.start_time}, events={self.events}) size={self.nBytes}"
