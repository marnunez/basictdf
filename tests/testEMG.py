from io import BytesIO
from tkinter import NS
from unittest import TestCase

import numpy as np
from basictdf.tdfEMG import EMGTrack, EMG
from basictdf.tdfTypes import Float32


class TestEMGTrack(TestCase):
    def test_creation(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        a = EMGTrack("Right Rectus Femoris", data)
        self.assertEqual(a.label, "Right Rectus Femoris")
        np.testing.assert_equal(a.data, data)
        self.assertEqual(a._segments, [slice(0, 11)])
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.nBytes, 256 + 4 + 4 + 4 + 4 + 11 * 4)

    def test_build(self):
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
        b += Float32.write(data)
        c = BytesIO(b)
        a = EMGTrack.build(c, 11)
        self.assertEqual(a.label, "Right Rectus Femoris")
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.nBytes, 256 + 4 + 4 + 4 + 4 + 11 * 4)
        self.assertEqual(a._segments, [slice(0, 11)])
        self.assertEqual(a.data.shape, (11,))
        d = BytesIO()
        a.write(d)
        self.assertEqual(d.getvalue(), b)

    def test_write(self):
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
        b += Float32.write(data)
        i = BytesIO(b)
        other = BytesIO()
        a.write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestEMG(TestCase):
    def test_creation(self):
        t1 = EMGTrack(
            "Right Rectus Femoris", np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        )
        t2 = EMGTrack(
            "Left Rectus Femoris",
            np.array([1.2, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11]),
        )
        a = EMG(frequency=1000, nSamples=11)
        self.assertEqual(a.nSamples, 11)
        self.assertEqual(a.frequency, 1000)
        self.assertEqual(a.nBytes, 16)
        self.assertEqual(a._signals, [])
        self.assertEqual(a._emgMap, [])

        # different nSamples
        with self.assertRaises(ValueError):
            less_samples = EMGTrack(
                "fasdf",
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
            


# class TestData3D(TestCase):
#     def test_creation(self):
#         t = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))

#         a = Data3D(
#             frequency=100,
#             nFrames=2,
#             volume=np.array([1, 2, 3]),
#             translationVector=np.array([1, 2, 3]),
#             rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         )
#         self.assertEqual(a.format, Data3dBlockFormat.byTrack)
#         self.assertEqual(a.nFrames, 2)
#         self.assertEqual(a.nTracks, 0)

#         # different nFrames
#         t2 = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
#         with self.assertRaises(ValueError):
#             a.add_track(t2)

#         with self.assertRaises(TypeError):
#             a.add_track([])

#         a.add_track(t)

#         self.assertEqual(a.nTracks, 1)
#         self.assertEqual(a.tracks, [t])

#         a.tracks = []
#         self.assertEqual(a.nTracks, 0)
#         a.tracks = [t, t]
#         self.assertEqual(a.nTracks, 2)

#     def test_build(self):
#         t = MarkerTrack("marker", np.array([[1, 2, 3], [4, 5, 6]]))
#         t2 = MarkerTrack("marker2", np.array([[4, 5, 6], [7, 8, 9]]))

#         dataBlock1 = Data3D(
#             frequency=100,
#             nFrames=2,
#             volume=np.array([1, 2, 3]),
#             translationVector=np.array([1, 2, 3]),
#             rotationMatrix=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         )

#         dataBlock1.tracks = [t, t2]

#         buff1 = BytesIO()
#         dataBlock1.write(buff1)
#         buff1.seek(0, 0)
#         dataBlock2 = Data3D.build(
#             buff1,
#             Data3dBlockFormat.byTrack,
#         )
#         buff2 = BytesIO()
#         dataBlock2.write(buff2)
#         self.assertEqual(dataBlock2.format, Data3dBlockFormat.byTrack)
#         self.assertEqual(dataBlock1, dataBlock2)
#         self.assertEqual(buff1.getvalue(), buff2.getvalue())
