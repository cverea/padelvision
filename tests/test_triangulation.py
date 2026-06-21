import cv2
import numpy as np
import pytest

from padelvision.geometry.triangulation import Triangulator, projection_matrix


def _project(P, points_3d):
    """Project (N,3) world points through a 3x4 matrix to (N,2) pixels."""
    hom = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    proj = (P @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


@pytest.fixture
def stereo_setup():
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    # Camera 1 at origin; camera 2 translated along X and slightly rotated.
    P1 = projection_matrix(K, np.eye(3), np.zeros(3))
    R = cv2.Rodrigues(np.array([0.0, 0.05, 0.0]))[0]
    T = np.array([-50.0, 0.0, 5.0])
    P2 = projection_matrix(K, R, T)
    return K, R, T, P1, P2


def test_triangulate_recovers_known_points(stereo_setup):
    _, _, _, P1, P2 = stereo_setup
    truth = np.array([[1.0, 2.0, 30.0], [-3.0, 5.0, 40.0], [0.5, -1.0, 25.0]])
    pts1 = _project(P1, truth)
    pts2 = _project(P2, truth)
    recovered = Triangulator(P1, P2).triangulate(pts1, pts2)
    np.testing.assert_allclose(recovered, truth, atol=1e-6)


def test_empty_input_returns_empty(stereo_setup):
    _, _, _, P1, P2 = stereo_setup
    out = Triangulator(P1, P2).triangulate(np.empty((0, 2)), np.empty((0, 2)))
    assert out.shape == (0, 3)


def test_mismatched_point_counts_raise(stereo_setup):
    _, _, _, P1, P2 = stereo_setup
    with pytest.raises(ValueError):
        Triangulator(P1, P2).triangulate([[1, 2]], [[1, 2], [3, 4]])


def test_bad_projection_shape_raises():
    with pytest.raises(ValueError):
        Triangulator(np.eye(3), np.eye(3))


def test_from_stereo_params_round_trip(tmp_path, stereo_setup):
    K, R, T, _, _ = stereo_setup
    path = tmp_path / "stereo_params.yml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("K1", K)
    fs.write("K2", K)
    fs.write("R", R)
    fs.write("T", T.reshape(3, 1))
    fs.release()

    triangulator = Triangulator.from_stereo_params(path)
    truth = np.array([[1.0, 2.0, 30.0], [-3.0, 5.0, 40.0]])
    pts1 = _project(triangulator.P1, truth)
    pts2 = _project(triangulator.P2, truth)
    np.testing.assert_allclose(triangulator.triangulate(pts1, pts2), truth, atol=1e-6)


def test_from_stereo_params_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Triangulator.from_stereo_params(tmp_path / "nope.yml")
