from io import BytesIO
from unittest import TestCase

import numpy as np

from basictdf.tdfEMG import EMG, EMGTrack
from basictdf.tdfTypes import f32


class TestEMGTrack(TestCase):
    def test_creation(self) -> None:
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        a = EMGTrack("Right Rectus Femoris", data)
        self.assertEqual(a.label, "Right Rectus Femoris")
        np.testing.assert_equal(a.data, data)
        self.assertEqual(a._segments, [slice(0, 11)])
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.nBytes, 256 + 4 + 4 + 4 + 4 + 11 * 4)

    def test_build(self) -> None:
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        b = b""
        # label
        b += b"Right Rectus Femoris" + b"\x00" * 236
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x0B\x00\x00\x00"
        # data
        b += f32.write(data)
        c = BytesIO(b)
        a = EMGTrack._build(c, 11)
        self.assertEqual(a.label, "Right Rectus Femoris")
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.nBytes, 256 + 4 + 4 + 4 + 4 + 11 * 4)
        self.assertEqual(a._segments, [slice(0, 11)])
        self.assertEqual(a.data.shape, (11,))
        d = BytesIO()
        a._write(d)
        self.assertEqual(d.getvalue(), b)

    def test_write(self) -> None:
        data = np.array([1, 2, 3.342, 4, 5, 6, 7, 8.54, 9, 10.123, 1e-3])
        a = EMGTrack("Right Rectus Femoris", data)
        self.assertEqual(len(a._segments), 1)
        self.assertEqual(a._segments[0], slice(0, 11))

        b = b""
        # label
        b += b"Right Rectus Femoris" + b"\x00" * 236
        # nSegments
        b += b"\x01\x00\x00\x00"
        # padding
        b += b"\x00\x00\x00\x00"
        # startFrame
        b += b"\x00\x00\x00\x00"
        # nFrames
        b += b"\x0B\x00\x00\x00"
        # data
        b += f32.write(data)
        i = BytesIO(b)
        other = BytesIO()
        a._write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestEMG(TestCase):
    def test_creation(self) -> None:
        a = EMG(frequency=1000, nSamples=11)
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.frequency, 1000)
        self.assertEqual(a.nBytes, 16)
        self.assertEqual(a._signals, [])
        self.assertEqual(a._emgMap, [])

        # different nSamples
        with self.assertRaises(ValueError):
            less_samples = EMGTrack(
                "I have less samples",
                np.array(
                    [
                        1,
                        2,
                        3,
                    ]
                ),
            )
            a.addSignal(less_samples)

        # different type
        with self.assertRaises(TypeError):
            a.addSignal("Not a EMGTrack")

    def test_addSignal(self) -> None:
        t1 = EMGTrack(
            "Right Rectus Femoris",
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        )
        t2 = EMGTrack(
            "Left Rectus Femoris",
            np.array([1.2, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11]),
        )
        a = EMG(frequency=1000, nSamples=11)
        a.addSignal(t1)
        self.assertEqual(a._signals, [t1])
        self.assertEqual(a._emgMap, [0])
        a.addSignal(t2)
        self.assertEqual(a._signals, [t1, t2])
        self.assertEqual(a._emgMap, [0, 1])

        a._signals = []
        a._emgMap = []

        a.addSignal(t1, 5)
        self.assertEqual(a._signals, [t1])
        self.assertEqual(a._emgMap, [5])

        a.addSignal(t2)
        self.assertEqual(a._signals, [t1, t2])
        self.assertEqual(a._emgMap, [5, 6])
