from __future__ import annotations

__doc__ = """
basictdf is a **read** and **write** parser for the BTS Bioengineering TDF file
format.
This format is tipically used as storage of raw data from a BTS motion capture
 acquisition system (e.g. raw EMG, 2D raw marker data) but can also serve as
storage for processed data
(e.g. 3D reconstructed marker data, filtered EMG signals, events).

## How to read a TDF file?

```python
from basictdf import Tdf

# You can use a context manager to automatically close the file
with Tdf("path/to/file.tdf") as tdf:
    data3D = tdf.data3D

# Or you can read individual blocks.
data3D = Tdf("path/to/file.tdf").data3D
```

## How to modify a TDF file?

The propper way to modify a TDF file is through a context manager.
This way, the file is guaranteed to be closed properly.
Specifying the mode as "r+b" (read and write) is also required
using the `basictdf.Tdf.allow_write` method.


You **can't** write to a tdf file like this.

```python
# Don't do this. This will fail
Tdf("path/to/file.tdf").data3D = oldData3D
```
Use a context manager instead

```python
# Do this instead
with Tdf("path/to/file.tdf").allow_write() as tdf:
    oldData3D = tdf.data3D

    # Let's add 1 to the X coordinate of the c7 marker
    oldData3D["c7"].X = oldData3D["c7"].X + 1

    tdf.data3D = oldData3D

# This is fine too
tdf = Tdf("path/to/file.tdf")

with tdf.allow_write() as tdf:
    tdf.data3D = oldData3D
```


## How to add a new block to a TDF file?
## How to create a TDF from scratch?

The easiest way to add a new block to a TDF is actually to modify an existing
block with the same type. However, if you want to add a completely new block,
 you can use any of the `basictdf.tdfBlock.Block` subclasses, like
 `basictdf.tdfData3D.Data3D` or `basictdf.tdfEvents.TemporalEventsData`.

```python
from basictdf import Tdf, Data3D, MarkerTrack
import numpy as np

# Create an empty data3D block
data3D = Data3D(
    frequency=1000,
    nFrames=1000,
    volume=np.array([1, 1, 1]),
    translationVector=np.array([0, 0, 0]),
    rotationMatrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

# Create a bogus marker track with random data
c7 = MarkerTrack("c7", np.random.rand(1000, 3))

# Add it to the data3D block
data3D.add_track(c7)

# Write it to a new TDF file
with Tdf.new("my_file.tdf").allow_write() as tdf:
    tdf.data3D = data3D

# This works too
tdf = Tdf.new("my_file.tdf")
with tdf.allow_write() as tdf:
    tdf.data3D = data3D
```
"""
from basictdf.basictdf import Tdf
from basictdf.tdfData3D import Data3D, MarkerTrack
from basictdf.tdfEMG import EMG, EMGTrack
from basictdf.tdfEvents import Event, EventsDataType, TemporalEventsData

__all__ = [
    "Tdf",
    "Data3D",
    "MarkerTrack",
    "TemporalEventsData",
    "Event",
    "EventsDataType",
    "EMG",
    "EMGTrack",
    "ForcePlatformsDataBlock",
    "ForcePlatformData",
]


__pdoc__ = {}
__pdoc__["basictdf.tdfUtils"] = False
__pdoc__["basictdf.tdfTypes"] = False
__pdoc__["collections.ABC"] = False
