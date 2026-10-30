import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np

from failurevla.action_buffer import ActionBuffer
from failurevla.data import load_episode
from failurevla.train import samples


class PipelineTest(unittest.TestCase):
    def test_trajectory_validation_and_history_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.h5"
            with h5py.File(path, "w") as f:
                f["t"] = [0., 1., 2., 3.]
                f["joints"] = np.tile(np.arange(4)[:, None], (1, 7))
                f["fingers"] = np.tile((np.arange(4)*200)[:, None], (1, 6))
                f["eef_left"] = np.zeros((4, 10))
                f["eef_right"] = np.zeros((4, 10))
                f["instruction"] = "test".encode()
            ep = load_episode(path)
            self.assertEqual(ep["joints"].shape, (4, 7))
            x, y = samples([path], 2, 1)
            self.assertEqual(x.shape, (2, 26))
            np.testing.assert_allclose(y[0, :7], 1)
            np.testing.assert_allclose(y[0, 7:], .1)
            with h5py.File(path, "r+") as f:
                f["t"][2] = 1
            with self.assertRaises(ValueError):
                load_episode(path)

    def test_buffer_rejects_stale_and_out_of_order_chunks(self):
        b = ActionBuffer(max_age_s=.2, max_state_error=.1)
        self.assertTrue(b.replace(2, 1., 1.1, [0], [.05], [[1], [2]]))
        self.assertFalse(b.replace(1, 1., 1.1, [0], [0], [[9]]))
        self.assertFalse(b.replace(3, 1., 1.3, [0], [0], [[9]]))
        self.assertFalse(b.replace(3, 1., 1.1, [0], [.2], [[9]]))
        np.testing.assert_array_equal(b.pop(), [1])
        np.testing.assert_array_equal(b.pop(), [2])
        self.assertIsNone(b.pop())
        self.assertTrue(b.replace(4, 2., 2.1, [0], [0], [[4]]))
        np.testing.assert_array_equal(b.pop(), [4])


if __name__ == "__main__":
    unittest.main()
