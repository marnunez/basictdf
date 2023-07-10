from io import BytesIO
from unittest import TestCase

import numpy as np

from basictdf.tdfData2D import Data2D, Data2DBlockFormat, Data2DFlags, Data2DPCK


class TestData2D(TestCase):
    def test_creation(self) -> None:
        data = np.empty((2, 2), dtype=object)
        data[0, 0] = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data[0, 1] = np.array([[5, 6], [7, 8]], dtype=np.float32)
        data[1, 0] = np.array([[9, 10], [11, 12]], dtype=np.float32)
        data[1, 1] = np.array([[13, 14], [15, 16]], dtype=np.float32)
        flags = Data2DFlags(0)
        startTime = 10.71131231312312
        d = Data2D(2, 2, 100, startTime, flags)
        d.data = data
        d._camMap = [0, 1]
        self.assertEqual(d.nCams, 2)
        self.assertEqual(d.nFrames, 2)
        self.assertEqual(d.frequency, 100)
        np.testing.assert_almost_equal(d.startTime, startTime)
        self.assertEqual(d.flags, flags)
        self.assertEqual(d.data.shape, (2, 2))
        for i in range(2):
            for j in range(2):
                np.testing.assert_equal(d.data[i, j], data[i, j])
        self.assertEqual(d.format, Data2DBlockFormat.PCKFormat)
        self.assertEqual(
            d.nBytes,
            4
            + 4
            + 4
            + 4
            + 4
            + 2 * 2
            + sum(
                data[i, j].nbytes
                for i in range(2)
                for j in range(2)
                if data[i, j] is not None
            )
            + 2 * 2 * 2,
        )

        c = BytesIO()
        d._write(c)
        c.seek(0, 0)
        new = Data2D._build(c, d.format)

        c.seek(0, 0)
        self.assertEqual(new.nBytes, len(c.getvalue()))

        self.assertEqual(d.nCams, new.nCams)
        self.assertEqual(d.nFrames, new.nFrames)
        self.assertEqual(d.frequency, new.frequency)
        np.testing.assert_almost_equal(d.startTime, new.startTime)
        self.assertEqual(d.flags, new.flags)
        for i in range(d.nCams):
            for j in range(d.nFrames):
                np.testing.assert_equal(d.data[i, j], new.data[i, j])
        self.assertEqual(d.format, new.format)
        self.assertEqual(d.nBytes, new.nBytes)


class TestData2dPCK(TestCase):
    def test_creation(self):
        nCameras = 2
        nFrames = 2

        data = np.empty((2, 2), dtype=object)
        data[0, 0] = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data[0, 1] = np.array([[5, 6], [7, 8]], dtype=np.float32)
        data[1, 0] = np.array([[9, 10], [11, 12]], dtype=np.float32)
        data[1, 1] = np.array([[13, 14], [15, 16]], dtype=np.float32)

        b = BytesIO()
        pack = Data2DPCK(data)
        pack._write(b)
        b.seek(0, 0)
        new_pack = Data2DPCK._build(b, nFrames, nCameras)
        c = BytesIO()
        new_pack._write(c)
        c.seek(0, 0)

        self.assertEqual(b.getvalue(), c.getvalue())
        self.assertEqual(pack.nBytes, len(b.getvalue()))
        self.assertEqual(new_pack.nBytes, len(c.getvalue()))
        self.assertEqual(pack.data.shape, new_pack.data.shape)
        for i in range(nCameras):
            for j in range(nFrames):
                np.testing.assert_equal(pack.data[i, j], new_pack.data[i, j])
