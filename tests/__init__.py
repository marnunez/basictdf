from __future__ import annotations
from pathlib import Path
from typing import Generator, Tuple

from basictdf.tdfData3D import Data3dBlockFormat

test_files_data = {
    "2838~aa~Walking 01.tdf": {
        "nBlocks": 8,
        "version": 1,
        "data3d": {
            "nFrames": 1123,
            "format": Data3dBlockFormat.byTrack,
            "frequency": 100,
            "startTime": 0,
            "nTracks": 22,
            "volume": [4.07, 1.76, 1.65],
            "rotationMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translationVector": [-1.165, 0.074, -0.418],
            "nLinks": 31,
            "nBytes": 234036,
        },
        "emg": {
            "nFrames": 710,
        },
    }
}


def test_file_feeder(
    block: str | None = None,
) -> Generator[Tuple[Path, dict], None, None]:
    tdf_files = (Path(__file__).parent / "test_files").glob("*.tdf")
    for tdf_file in tdf_files:
        tdf_metadata = test_files_data.get(tdf_file.name, {})
        yield tdf_file, tdf_metadata
