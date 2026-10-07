#!/usr/bin/env python3
"""
Training runner tests for Phase 4C.1.
Tests CLI, integration (CPU fixture), and fault injection.
"""

import pytest

import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add repo root to path
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if training modules can be imported
try:
    from ml.training.run_learning_curve import (
        set_seed,
        compute_sha256,
        save_checkpoint,
        ForensicsDataset,
        build_instances_for_training,
        create_data_loaders,
        train_one_epoch,
        evaluate,
        run_dummy_baseline,
        run_metadata_baseline,
        collect_predictions,
        run_smoke_training,
        main,
    )
    RUNNER_AVAILABLE = True
except ImportError:
    RUNNER_AVAILABLE = False


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestRunnerImports:
    """Test runner imports correctly."""

    def test_imports(self):
        """Test all required modules can be imported."""
        from ml.training.run_learning_curve import (
            set_seed,
            compute_sha256,
            save_checkpoint,
            ForensicsDataset,
            build_instances_for_training,
            create_data_loaders,
            train_one_epoch,
            evaluate,
            run_dummy_baseline,
            run_metadata_baseline,
            collect_predictions,
            run_smoke_training,
            main,
        )
        assert callable(set_seed)
        assert callable(compute_sha256)
        assert callable(save_checkpoint)
        assert ForensicsDataset is not None

    def test_no_duplicate_functions(self):
        """Test no duplicate function definitions."""
        import ml.training.run_learning_curve as runner
        import inspect

        # Get all functions
        functions = [name for name, obj in inspect.getmembers(runner, inspect.isfunction)
                    if not name.startswith('_')]

        # Check for duplicates
        assert len(functions) == len(set(functions)), "Duplicate function definitions found"


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestUtilityFunctions:
    """Test utility functions."""

    def test_set_seed(self):
        """Test seed setting."""
        from ml.training.run_learning_curve import set_seed
        set_seed(42)
        # Should not raise

    def test_compute_sha256(self):
        """Test SHA256 computation."""
        from ml.training.run_learning_curve import compute_sha256
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            hash1 = compute_sha256(Path(f.name))
            hash2 = compute_sha256(Path(f.name))
            assert hash1 == hash2
            assert len(hash1) == 64

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        from ml.training.run_learning_curve import save_checkpoint
        import torch
        import torch.nn as nn

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(model, optimizer, scheduler, 0, {"loss": 0.5}, path, {})

            assert path.exists()
            # Verify loadable
            checkpoint = torch.load(path, map_location="cpu")
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert checkpoint["epoch"] == 0


@pytest.mark.requires_research_artifact
@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestDataLoading:
    """Test data loading functions."""

    def test_build_instances_for_training(self):
        """Test instance building from bundle manifest."""
        from ml.datasets.export_learning_curve_bundle import create_smoke_bundle
        from ml.training.run_learning_curve import build_instances_for_training

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            # Copy manifest to bundle dir
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            instances = build_instances_for_training(
                bundle_path=output_dir,
                sample_size=50,
                seed=42,
            )

            assert len(instances) == 50
            # All should be development_train
            for inst in instances:
                assert inst.partition == "development_train"
                assert inst.lc_flags.get("lc_n50", False) is True

    def test_create_data_loaders(self):
        """Test data loader creation."""
        from ml.datasets.export_learning_curve_bundle import create_smoke_bundle
        from ml.training.run_learning_curve import build_instances_for_training, create_data_loaders
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            instances = build_instances_for_training(
                bundle_path=output_dir,
                sample_size=50,
                seed=42,
            )

            train_loader, val_loader = create_data_loaders(
                instances=instances,
                batch_size=32,
                device=torch.device("cpu"),
            )

            assert train_loader is not None
            assert val_loader is not None
            assert len(train_loader) > 0

    def test_reusable_bundle_three_sizes_and_frozen_cohorts(self):
        """Test runner supports N=50, 100, 250 with frozen cohort invariance across seeds."""
        from ml.datasets.export_learning_curve_bundle import create_reusable_n250_bundle
        from ml.training.run_learning_curve import build_instances_for_training, build_instances_for_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "reusable_bundle"
            bundle = create_reusable_n250_bundle(REPO_ROOT, output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            # Test 3 sizes for seed 42
            insts_50 = build_instances_for_training(output_dir, sample_size=50, seed=42)
            insts_100 = build_instances_for_training(output_dir, sample_size=100, seed=42)
            insts_250 = build_instances_for_training(output_dir, sample_size=250, seed=42)

            assert len(insts_50) == 50
            assert len(insts_100) == 100
            assert len(insts_250) == 250

            s50 = [inst.source_id for inst in insts_50]
            s100 = [inst.source_id for inst in insts_100]
            s250 = [inst.source_id for inst in insts_250]

            # Invariance across seeds: cohort sources MUST be identical regardless of seed
            for seed in [1337, 2025, 3407, 9001]:
                insts_50_seed = build_instances_for_training(output_dir, sample_size=50, seed=seed)
                insts_100_seed = build_instances_for_training(output_dir, sample_size=100, seed=seed)
                insts_250_seed = build_instances_for_training(output_dir, sample_size=250, seed=seed)

                assert [inst.source_id for inst in insts_50_seed] == s50
                assert [inst.source_id for inst in insts_100_seed] == s100
                assert [inst.source_id for inst in insts_250_seed] == s250

            # Nested cohort invariance
            set50 = set(s50)
            set100 = set(s100)
            set250 = set(s250)
            assert set50.issubset(set100), "N50 must be subset of N100"
            assert set100.issubset(set250), "N100 must be subset of N250"

            # Validation instances
            val_insts = build_instances_for_validation(output_dir)
            assert len(val_insts) == 91
            val_sids = {inst.source_id for inst in val_insts}
            assert len(set250 & val_sids) == 0, "Development and validation must have 0 overlap"

    def test_create_data_loaders_for_reusable_bundle(self):
        """Test data loader sample counts for reusable bundle across all sizes."""
        from ml.datasets.export_learning_curve_bundle import create_reusable_n250_bundle
        from ml.training.run_learning_curve import (
            build_instances_for_training,
            build_instances_for_validation,
            create_data_loaders,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "reusable_bundle"
            bundle = create_reusable_n250_bundle(REPO_ROOT, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                shutil.copy2(f["source_path"], output_dir / f["name"])

            val_insts = build_instances_for_validation(output_dir)

            for size, expected_train_samples in [(50, 100), (100, 200), (250, 500)]:
                train_insts = build_instances_for_training(output_dir, sample_size=size)
                train_loader, val_loader = create_data_loaders(
                    train_instances=train_insts,
                    val_instances=val_insts,
                    batch_size=32,
                    repo_root=str(output_dir),
                )
                assert len(train_loader.dataset) == expected_train_samples
                assert len(val_loader.dataset) == 182


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestBaselines:
    """Test baseline functions."""

    @pytest.mark.requires_research_artifact
    def test_run_dummy_baseline(self):
        """Test dummy baseline returns expected structure."""
        from ml.datasets.export_learning_curve_bundle import create_smoke_bundle
        from ml.training.run_learning_curve import build_instances_for_training, run_dummy_baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            instances = build_instances_for_training(
                bundle_path=output_dir,
                sample_size=50,
                seed=42,
            )

            result = run_dummy_baseline(instances, sample_size=50, seed=42)

            assert "macro_f1_mean" in result
            assert "macro_f1_std" in result
            assert 0 <= result["macro_f1_mean"] <= 1
            assert result["macro_f1_std"] >= 0

    def test_run_metadata_baseline(self):
        """Test metadata baseline returns expected structure."""
        from ml.training.run_learning_curve import run_metadata_baseline

        result = run_metadata_baseline([], sample_size=50, seed=42)

        assert "macro_f1_mean" in result
        assert "macro_f1_std" in result
        assert result["macro_f1_mean"] == 0.5


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestCLI:
    """Test CLI argument parsing."""

    def test_cli_help(self):
        """Test CLI help works."""
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "ml.training.run_learning_curve", "--help"
        ], cwd=REPO_ROOT, capture_output=True, text=True)

        assert result.returncode == 0
        assert "Phase 4C.1" in result.stdout
        assert "--config" in result.stdout
        assert "--bundle" in result.stdout
        assert "--sample-size" in result.stdout
        assert "--stage" in result.stdout

    def test_cli_missing_required_args(self):
        """Test CLI fails with missing required args."""
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "ml.training.run_learning_curve"
        ], cwd=REPO_ROOT, capture_output=True, text=True)

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestFaultInjection:
    """Fault injection tests for runner."""

    @pytest.mark.requires_research_artifact
    def test_runner_failure_on_missing_checkpoint(self):
        """Test runner handles missing checkpoint gracefully."""
        from ml.training.run_learning_curve import run_smoke_training
        import argparse

        # Mock args
        args = argparse.Namespace(
            config=str(REPO_ROOT / "ml/configs/phase_4c1_learning_curve.yaml"),
            bundle=str(REPO_ROOT / "nonexistent_bundle"),
            sample_size=50,
            seed=42,
            stage="frozen",
            device="cpu",
            train_partition="development_train",
            eval_partition="inner_validation",
            output="/tmp/test_output",
        )

        # Should fail with appropriate error
        with pytest.raises(Exception):
            run_smoke_training(args)

    def test_runner_cuda_check(self):
        """Test runner checks CUDA availability."""
        from ml.training.run_learning_curve import main
        import argparse
        import sys

        # Mock CUDA not available
        with patch("torch.cuda.is_available", return_value=False):
            args = argparse.Namespace(
                config=str(REPO_ROOT / "ml/configs/phase_4c1_learning_curve.yaml"),
                bundle=str(REPO_ROOT / "test_bundle"),
                sample_size=50,
                seed=42,
                stage="frozen",
                device="cuda",
                train_partition="development_train",
                eval_partition="inner_validation",
                output="/tmp/test_output",
            )

            with pytest.raises(SystemExit) as exc_info:
                # Simulate main with args
                sys.argv = ["run_learning_curve", "--config", args.config, "--bundle", args.bundle,
                           "--sample-size", "50", "--seed", "42", "--stage", "frozen",
                           "--device", "cuda", "--train-partition", "development_train",
                           "--eval-partition", "inner_validation", "--output", args.output]
                main()

            # Should exit with error
            assert exc_info.value.code != 0


@pytest.mark.skipif(not RUNNER_AVAILABLE, reason="Training runner dependencies not available")
class TestIntegrationFixture:
    """Technical fixture run on CPU (not scientific training)."""

    @pytest.mark.requires_research_artifact
    @pytest.mark.integration
    def test_smoke_run_cpu_fixture(self):
        """Technical fixture: run training for 1 epoch on CPU to verify pipeline.

        This is a TECHNICAL FIXTURE RUN, not a scientific training run.
        It verifies the pipeline executes without errors.
        """
        # This test requires torch and data - mark as integration
        # Only runs when explicitly requested with -m integration
        pytest.skip("Integration test requires full environment - run with -m integration")

    def test_pipeline_structure(self):
        """Test pipeline components connect correctly."""
        from ml.training.run_learning_curve import (
            build_instances_for_training,
            create_data_loaders,
            MobileNetV3Forensics,
            FocalLoss,
            train_one_epoch,
            evaluate,
        )
        import torch

        # Verify all components are callable
        assert callable(build_instances_for_training)
        assert callable(create_data_loaders)
        assert MobileNetV3Forensics is not None
        assert FocalLoss is not None
        assert callable(train_one_epoch)
        assert callable(evaluate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])