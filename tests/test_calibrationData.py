from io import BytesIO
from unittest import TestCase

import numpy as np

from basictdf.tdfCalibrationData import (
    CalibrationDataBlock,
    CalibrationDataBlockFormat,
    DistorsionModel,
    SeelabCameraData,
    CameraViewPort,
)


class TestCalibrationData(TestCase):
    def test_seelab_camera_data(self):
        rotation_matrix = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.float32,
        )
        translation_vector = np.array(
            [1, 2, 3],
            dtype=np.float32,
        )
        focus = np.array(
            [1, 2],
            dtype=np.float32,
        )
        optical_center = np.array(
            [1, 2],
            dtype=np.float32,
        )
        radial_distortion = np.array(
            [3, 4],
            dtype=np.float32,
        )
        decentering = np.array(
            [5, 6],
            dtype=np.float32,
        )
        thin_prism = np.array(
            [7, 8],
            dtype=np.float32,
        )
        view_port = CameraViewPort(
            origin=np.array(
                [1, 2],
                dtype=np.float32,
            ),
            size=np.array(
                [3, 4],
                dtype=np.float32,
            ),
        )

        cam = SeelabCameraData(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            focus=focus,
            optical_center=optical_center,
            radial_distortion=radial_distortion,
            decentering=decentering,
            thin_prism=thin_prism,
            view_port=view_port,
        )

        b = BytesIO()
        cam._write(b)
        b.seek(0, 0)

        new_cam = SeelabCameraData._build(b)

        np.testing.assert_equal(new_cam.rotation_matrix, rotation_matrix)
        np.testing.assert_equal(new_cam.translation_vector, translation_vector)
        np.testing.assert_equal(new_cam.focus, focus)
        np.testing.assert_equal(new_cam.optical_center, optical_center)
        np.testing.assert_equal(new_cam.radial_distortion, radial_distortion)
        np.testing.assert_equal(new_cam.decentering, decentering)
        np.testing.assert_equal(new_cam.thin_prism, thin_prism)
        self.assertEqual(new_cam.view_port, view_port)

        c = BytesIO()
        new_cam._write(c)
        c.seek(0, 0)
        b.seek(0, 0)

        self.assertEqual(c.read(), b.read())

    def test_creation(self):
        return
        n_cams = 2
        distortion_model = DistorsionModel.AmassDistorsion
        calibration_volume = np.array(
            [
                [1, 2, 3],
            ],
            dtype=np.float32,
        )
        rotation_matrix = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.float32,
        )
        translation_vector = np.array(
            [1, 2, 3],
            dtype=np.float32,
        )
        calibration_map = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=np.float32,
        )
        block = CalibrationDataBlock(
            distorsion_model=distortion_model,
            calibration_volume=calibration_volume,
            calibration_volume_rotation_matrix=rotation_matrix,
            calibration_volume_translation_vector=translation_vector,
            cameras_calibration_map=calibration_map,
            cam_data=1,
        )
