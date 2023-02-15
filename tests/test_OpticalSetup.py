from io import BytesIO
from unittest import TestCase

import numpy as np
from basictdf.tdfOpticalSystem import (
    CameraViewPort,
    OpticalSetupBlock,
    OpticalSetupBlockFormat,
    OpticalChannelData,
)


class TestChannel(TestCase):
    def test_creation(self):
        a = OpticalChannelData(
            camera_name="Camera",
            camera_type="This type",
            camera_viewport=np.array([(0, 0), (1, 1)]),
            lens_name="Lens",
            logical_camera_index=1,
        )
        self.assertEqual(a.camera_name, "Camera")
        self.assertEqual(a.camera_type, "This type")
        self.assertEqual(a.lens_name, "Lens")
        self.assertEqual(a.logical_camera_index, 1)
        self.assertEqual(
            a.camera_viewport, CameraViewPort(np.array((0, 0)), np.array((1, 1)))
        )
        cv = CameraViewPort(origin=np.array((0, 1)), size=np.array((1, 1)))
        a = OpticalChannelData(
            camera_name="Camera",
            camera_type="This type",
            camera_viewport=cv,
            lens_name="Lens",
            logical_camera_index=1,
        )
        self.assertEqual(a.camera_viewport, cv)

    def test_build(self):
        b = b""
        # logical index
        b += b"\x01\x00\x00\x00"
        # Reserved
        b += b"\x00\x00\x00\x00"
        # lens name
        b += b"Lens" + b"\x00" * 28
        # camera type
        b += b"This type" + b"\x00" * 23
        # camera name
        b += b"Camera" + b"\x00" * 26
        # camera viewport
        vp = CameraViewPort(origin=(0, 0), size=(1, 1))
        b += vp.write()
        c = BytesIO(b)
        a = OpticalChannelData._build(c)
        self.assertEqual(a.camera_name, "Camera")
        self.assertEqual(a.camera_type, "This type")
        self.assertEqual(a.lens_name, "Lens")
        self.assertEqual(a.logical_camera_index, 1)
        np.testing.assert_equal(a.camera_viewport, vp)
        d = BytesIO()
        a._write(d)
        self.assertEqual(d.getvalue(), b)

    def test_write(self):
        vp = CameraViewPort(origin=(0, 0), size=(1, 1))
        a = OpticalChannelData(
            camera_name="Camera",
            camera_type="This type",
            lens_name="Lens",
            logical_camera_index=1,
            camera_viewport=vp,
        )

        b = b""
        # logical index
        b += b"\x01\x00\x00\x00"
        # Reserved
        b += b"\x00\x00\x00\x00"
        # lens name
        b += b"Lens" + b"\x00" * 28
        # camera type
        b += b"This type" + b"\x00" * 23
        # camera name
        b += b"Camera" + b"\x00" * 26
        # camera viewport
        b += vp.write()

        i = BytesIO(b)
        other = BytesIO()
        a._write(other)
        self.assertEqual(i.getvalue(), other.getvalue())
        self.assertEqual(a.nBytes, len(b))


class TestOpticalSetupBlock(TestCase):
    def test_creation(self):
        nChannels = 8
        channels = [
            OpticalChannelData(
                camera_name=f"Camera {i}",
                camera_type=f"Type n° {i}",
                lens_name=f"Lens {i}",
                logical_camera_index=i,
                camera_viewport=CameraViewPort(origin=(0, 0), size=(i, i)),
            )
            for i in range(nChannels)
        ]
        a = OpticalSetupBlock(channels=channels)

        self.assertEqual(a.format, OpticalSetupBlockFormat.basicFormat)
        self.assertEqual(a.channels, channels)
        self.assertEqual(a.channels[0].camera_name, "Camera 0")
        self.assertEqual(a.channels[0].camera_type, "Type n° 0")
        self.assertEqual(a.channels[0].lens_name, "Lens 0")
        self.assertEqual(a.channels[0].logical_camera_index, 0)
        self.assertEqual(a.channels[0].camera_viewport, CameraViewPort((0, 0), (0, 0)))

    def test_build(self):
        nChannels = 8
        channels = [
            OpticalChannelData(
                camera_name=f"Camera {i}",
                camera_type=f"Type n° {i}",
                lens_name=f"Lens {i}",
                logical_camera_index=i,
                camera_viewport=CameraViewPort(origin=(0, 0), size=(i, i)),
            )
            for i in range(nChannels)
        ]
        block1 = OpticalSetupBlock(channels=channels)

        buff = BytesIO()
        block1._write(buff)
        buff.seek(0, 0)
        block2 = OpticalSetupBlock._build(buff, format=block1.format)
        buff2 = BytesIO()
        block2._write(buff2)

        self.assertEqual(block2.format, OpticalSetupBlockFormat.basicFormat)
        self.assertEqual(block2.channels, channels)
        self.assertEqual(block2.channels[0].camera_name, "Camera 0")
        self.assertEqual(block2.channels[0].camera_type, "Type n° 0")
        self.assertEqual(block2.channels[0].lens_name, "Lens 0")
        self.assertEqual(block2.channels[0].logical_camera_index, 0)
        self.assertEqual(
            block2.channels[0].camera_viewport, CameraViewPort((0, 0), (0, 0))
        )
        self.assertEqual(buff.getvalue(), buff2.getvalue())
        self.assertEqual(block1.nBytes, len(buff.getvalue()))


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
#         dataBlock1._write(buff1)
#         buff1.seek(0, 0)
#         dataBlock2 = Data3D._build(
#             buff1,
#             Data3dBlockFormat.byTrack,
#         )
#         buff2 = BytesIO()
#         dataBlock2._write(buff2)
#         self.assertEqual(dataBlock2.format, Data3dBlockFormat.byTrack)
#         self.assertEqual(dataBlock1, dataBlock2)
#         self.assertEqual(dataBlock1.tracks, dataBlock2.tracks)
#         self.assertEqual(buff1.getvalue(), buff2.getvalue())
#         self.assertEqual(dataBlock1.nBytes, len(buff2.getvalue()))
