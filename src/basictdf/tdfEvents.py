__doc__ = "Events data module."

from enum import Enum
from typing import Iterator, Union

import numpy as np

from basictdf.tdfBlock import Block, BlockType
from basictdf.tdfTypes import BTSString, f32, i32, u32
from basictdf.tdfUtils import is_iterable


class TemporalEventsDataFormat(Enum):
    unknown = 0
    standard = 1


class EventsDataType(Enum):
    singleEvent = 0
    eventSequence = 1


class Event:
    """
    A class representing a single event or a sequence of events.
    """

    def __init__(self, label, values=[], type=EventsDataType.singleEvent) -> None:
        self.label = label
        self.type = type
        if not is_iterable(values):
            raise TypeError("Values must be iterable")
        if isinstance(values, np.ndarray) and values.dtype == np.dtype("<f4"):
            self.values = values
        else:
            self.values = np.array(values, dtype="<f4")

        if len(values) > 1 and type == EventsDataType.singleEvent:
            raise TypeError("Can't have more than one value for a single event")

    def _write(self, stream) -> None:
        BTSString.bwrite(stream, 256, self.label)
        u32.bwrite(stream, self.type.value)  # type
        u32.bwrite(stream, len(self.values))  # nItems
        f32.bwrite(stream, self.values)

    @staticmethod
    def _build(stream) -> "Event":
        label = BTSString.bread(stream, 256)
        type_ = EventsDataType(u32.bread(stream))
        nItems = i32.bread(stream)
        values = np.array([f32.bread(stream) for _ in range(nItems)])
        return Event(label, values, type_)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def nBytes(self) -> int:
        return 256 + 4 + 4 + len(self.values) * 4

    def __repr__(self) -> str:
        return f"Event(label={self.label}, type={self.type}, items={self.values})"


class TemporalEventsData(Block):
    """
    A class to represent a TDF temporal events data block.
    """

    type = BlockType.temporalEventsData

    def __init__(self, format=TemporalEventsDataFormat.standard, start_time=0.0):
        super().__init__()
        self.format = format
        self.start_time = start_time
        self.events = []

    @staticmethod
    def _build(stream, format) -> "TemporalEventsData":
        format = TemporalEventsDataFormat(format)
        nEvents = i32.bread(stream)
        start_time = f32.bread(stream)

        t = TemporalEventsData(format, start_time)
        t.events = [Event._build(stream) for _ in range(nEvents)]

        return t

    def _write(self, stream) -> None:
        i32.bwrite(stream, len(self.events))
        f32.bwrite(stream, self.start_time)
        for event in self.events:
            event._write(stream)

    @property
    def nBytes(self) -> int:
        return 4 + 4 + sum(i.nBytes for i in self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, item: Union[int, str]) -> Event:
        if isinstance(item, int):
            return self.events[item]
        elif isinstance(item, str):
            try:
                return next(e for e in self.events if e.label == item)
            except StopIteration:
                raise KeyError(f"Event with label {item} not found")
        raise TypeError(f"Invalid key type: {type(item)}")

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __contains__(self, value: Union[Event, str]) -> bool:
        if isinstance(value, Event):
            return value in self.events
        elif isinstance(value, str):
            return any(value == event.label for event in self.events)
        raise TypeError(f"Invalid key type: {type(value)}")

    def __repr__(self) -> str:
        return (
            f"TemporalEventsData(format={self.format}, nEvents={len(self.events)}, "
            f"start_time={self.start_time}, events={self.events}) size={self.nBytes}"
        )
