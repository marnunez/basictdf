from io import BytesIO
from unittest import TestCase
from datetime import datetime

import numpy as np

from basictdf.tdfData2D import Data2D, Data2DBlockFormat, Data2DFlags


class TestData2D(TestCase):
    def test_creation(self) -> None:
        data = np.empty((2, 2), dtype=object)
        data[0, 0] = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data[0, 1] = np.array([[5, 6], [7, 8]], dtype=np.float32)
        data[1, 0] = np.array([[9, 10], [11, 12]], dtype=np.float32)
        data[1, 1] = np.array([[13, 14], [15, 16]], dtype=np.float32)
        flags = Data2DFlags(0)
        # Today without milliseconds
        startTime = datetime.now().replace(microsecond=0)
        d = Data2D(2, 2, 100, startTime, flags, data)
        d._camMap = [0, 1]
        self.assertEqual(d.nCams, 2)
        self.assertEqual(d.nFrames, 2)
        self.assertEqual(d.frequency, 100)
        self.assertEqual(d.startTime, startTime)
        self.assertEqual(d.flags, flags)
        np.testing.assert_equal(d.data, data)
        self.assertEqual(d.format, Data2DBlockFormat.PCKFormat)
        self.assertEqual(d.nBytes, 4 + 4 + 4 + 4 + 4 + 2 * 2 + data.nbytes)

        c = BytesIO()
        d._write(c)
        c.seek(0, 0)
        new = Data2D._build(c, d.format)

        self.assertEqual(d.nCams, new.nCams)
        self.assertEqual(d.nFrames, new.nFrames)
        self.assertEqual(d.frequency, new.frequency)
        self.assertEqual(d.startTime, new.startTime)
        self.assertEqual(d.flags, new.flags)
        for i in range(d.nCams):
            for j in range(d.nFrames):
                np.testing.assert_equal(d.data[i, j], new.data[i, j])
