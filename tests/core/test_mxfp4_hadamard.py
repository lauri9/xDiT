import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from xfuser.core.quant.activation_stats import (
    compute_mx_group_stats,
    maybe_log_activation_stats,
    should_log_activation_stats,
)
from xfuser.core.quant.hadamard import (
    build_hadamard_matrix,
    fold_weight_for_hadamard,
    hadamard_rotate,
)


class TestHadamardRotate(unittest.TestCase):
    def test_orthonormal_rotation_preserves_norm_per_block(self):
        block_r = 32
        R = build_hadamard_matrix(block_r, dtype=torch.float32)
        identity = torch.matmul(R, R.transpose(-1, -2))
        self.assertTrue(torch.allclose(identity, torch.eye(block_r), atol=1e-5, rtol=1e-5))

    def test_fold_preserves_linear_output(self):
        torch.manual_seed(0)
        block_r = 32
        in_features = 128
        out_features = 64
        batch = 8
        x = torch.randn(batch, in_features, dtype=torch.float32)
        w = torch.randn(out_features, in_features, dtype=torch.float32)

        w_folded = fold_weight_for_hadamard(w, block_r)
        R = build_hadamard_matrix(block_r, dtype=torch.float32)
        x_rot = hadamard_rotate(x, R)

        y_ref = x @ w.transpose(-1, -2)
        y_rot = x_rot @ w_folded.transpose(-1, -2)
        self.assertTrue(torch.allclose(y_ref, y_rot, atol=1e-2, rtol=1e-2))


class TestActivationStats(unittest.TestCase):
    def test_concentrated_activations_have_higher_outlier_ratio(self):
        group_size = 32
        num_groups = 4
        concentrated = torch.zeros(4, group_size * num_groups)
        for g in range(num_groups):
            concentrated[:, g * group_size] = 10.0

        spread = torch.randn(4, group_size * num_groups)
        R = build_hadamard_matrix(group_size, dtype=torch.float32)
        rotated = hadamard_rotate(concentrated, R)

        concentrated_stats = compute_mx_group_stats(concentrated, group_size=group_size)
        rotated_stats = compute_mx_group_stats(rotated, group_size=group_size)
        spread_stats = compute_mx_group_stats(spread, group_size=group_size)

        self.assertGreater(
            concentrated_stats["group_max_over_rms_mean"],
            rotated_stats["group_max_over_rms_mean"],
        )
        self.assertGreater(
            concentrated_stats["group_max_over_rms_mean"],
            spread_stats["group_max_over_rms_mean"] * 0.5,
        )

    @patch.dict(os.environ, {"XFUSER_MXFP4_LOG_ACTIVATION_STATS": "1", "LOCAL_RANK": "0"}, clear=False)
    def test_logging_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "stats.csv")
            with patch.dict(
                os.environ,
                {"XFUSER_MXFP4_LOG_ACTIVATION_STATS_OUTPUT": csv_path},
                clear=False,
            ):
                x = torch.randn(8, 64)
                maybe_log_activation_stats(
                    layer_name="test.layer",
                    x=x,
                    group_size=32,
                    phase="pre_hadamard",
                    hadamard_enabled=False,
                    step=0,
                )
                self.assertTrue(os.path.exists(csv_path))
                with open(csv_path, encoding="utf-8") as handle:
                    contents = handle.read()
                self.assertIn("group_max_over_rms_mean", contents)
                self.assertIn("test.layer", contents)


class TestMxfp4Defaults(unittest.TestCase):
    def test_hadamard_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XFUSER_AITER_MXFP4_HADAMARD", None)
            from xfuser.envs import environment_variables, parse_env_bool

            self.assertFalse(
                parse_env_bool(environment_variables["AITER_MXFP4_HADAMARD"](), default=False)
            )

    def test_activation_stats_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XFUSER_MXFP4_LOG_ACTIVATION_STATS", None)
            self.assertFalse(should_log_activation_stats())


if __name__ == "__main__":
    unittest.main()
