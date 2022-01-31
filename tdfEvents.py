from enum import Enum
from tdfBlock import Block
from io import BytesIO
import struct
from tdfUtils import BTSString, is_iterable


class TemporalEventsDataFormat(Enum):
    unknown = 0
    standard = 1


class EventsDataType(Enum):
    singleEvent = 0
    eventSequence = 1


class Event:
    def __init__(self, label, values, type=None):
        self.label = label
        self.values = values
        if not is_iterable(values):
            raise TypeError("Values must be iterable")

        if not type:
            if len(values) == 1:
                self.type = EventsDataType.singleEvent
            else:
                self.type = EventsDataType.eventSequence
        else:
            self.type = type

    def write(self):
        data = b""
        data += BTSString.write(256, self.label)  # label
        data += struct.pack("<I", self.type.value)  # type
        data += struct.pack("<I", len(self.values))  # number of values
        data += struct.pack(f"<{len(self.values)}f", *self.values)  # values
        return data

    def __len__(self):
        return 256 + 4 + 4 + len(self.values) * 4

    def __repr__(self):
        return f"Event(label={self.label}, type={self.type}, items={self.values})"


class TemporalEventsData(Block):
    def __init__(self, entry, block_data):
        super().__init__(entry, block_data)
        self.format = TemporalEventsDataFormat(entry["format"])
        self.nEvents = struct.unpack("<i", self.data.read(4))[0]
        self.start_time = struct.unpack("<f", self.data.read(4))[0]
        self.events = []
        for i in range(self.nEvents):
            label = BTSString.read(256, self.data.read(256))
            type = EventsDataType(struct.unpack("<I", self.data.read(4))[0])
            nItems = struct.unpack("<i", self.data.read(4))[0]
            items = [struct.unpack("<f", self.data.read(4))[0] for _ in range(nItems)]
            self.events.append(Event(label, items, type))

    def write(self):
        data = b""
        data += struct.pack("<I", self.format.value)
        data += struct.pack("<i", self.nEvents)
        data += struct.pack("<f", self.start_time)
        for event in self.events:
            data += event.write()
        return data

    def __len__(self):
        return 4 + 4 + sum(len(i) for i in self.events)

    def __repr__(self):
        return f"TemporalEventsData(format={self.format}, nEvents={self.nEvents}, start_time={self.start_time}, events={self.events}) size={self.size}"
