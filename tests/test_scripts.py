"""
Tests for setup scripts.

These tests verify that the data setup scripts work correctly
without actually downloading the full dataset.
"""

import os
import subprocess
from pathlib import Path

import pytest


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATABASE_DIR = PROJECT_ROOT / "database"


class TestRunFullExperimentScript:
    """Tests for scripts/run_full_experiment.sh"""

    @pytest.fixture
    def script_path(self) -> Path:
        """Get path to run_full_experiment.sh"""
        return SCRIPTS_DIR / "run_full_experiment.sh"

    def test_script_exists(self, script_path: Path) -> None:
        """Test that run_full_experiment.sh exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_executable(self, script_path: Path) -> None:
        """Test that run_full_experiment.sh is executable."""
        assert os.access(script_path, os.X_OK), f"Script is not executable: {script_path}"

    def test_script_has_shebang(self, script_path: Path) -> None:
        """Test that script has proper shebang."""
        with open(script_path, "r") as f:
            first_line = f.readline()
        assert first_line.startswith("#!/bin/bash"), "Script should start with #!/bin/bash"

    def test_script_references_train_scripts(self, script_path: Path) -> None:
        """Test that script calls both training scripts."""
        content = script_path.read_text()
        assert "train_tsm.py" in content, "Script should call train_tsm.py"
        assert "train_ffn.py" in content, "Script should call train_ffn.py"

    def test_script_has_required_ffn_args(self, script_path: Path) -> None:
        """Test that script passes required args to train_ffn.py."""
        content = script_path.read_text()
        assert "--video_dir" in content, "Script should pass --video_dir to train_ffn.py"
        assert "--labels_dir" in content, "Script should pass --labels_dir to train_ffn.py"

    def test_training_scripts_exist(self) -> None:
        """Test that the training scripts referenced by the experiment script exist."""
        train_tsm = PROJECT_ROOT / "train_tsm.py"
        train_ffn = PROJECT_ROOT / "train_ffn.py"
        assert train_tsm.exists(), f"train_tsm.py not found: {train_tsm}"
        assert train_ffn.exists(), f"train_ffn.py not found: {train_ffn}"


class TestSetupDataScript:
    """Tests for scripts/setup_data.sh"""

    @pytest.fixture
    def script_path(self) -> Path:
        """Get path to setup_data.sh"""
        return SCRIPTS_DIR / "setup_data.sh"

    def test_script_exists(self, script_path: Path) -> None:
        """Test that setup_data.sh exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_executable(self, script_path: Path) -> None:
        """Test that setup_data.sh is executable."""
        assert os.access(script_path, os.X_OK), f"Script is not executable: {script_path}"

    def test_script_help_flag(self, script_path: Path) -> None:
        """Test that --help flag works."""
        result = subprocess.run(
            [str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "Usage" in result.stdout, "Help output should contain 'Usage'"

    def test_script_has_shebang(self, script_path: Path) -> None:
        """Test that script has proper shebang."""
        with open(script_path, "r") as f:
            first_line = f.readline()
        assert first_line.startswith("#!/bin/bash"), "Script should start with #!/bin/bash"


class TestLabelsDirectory:
    """Tests for labels directory structure (labels downloaded on cluster)."""

    def test_labels_directory_exists(self) -> None:
        """Test that labels directory exists (files downloaded on cluster)."""
        labels_dir = DATABASE_DIR / "labels"
        assert labels_dir.exists(), f"Labels directory not found: {labels_dir}"

    def test_gitkeep_exists(self) -> None:
        """Test that .gitkeep exists to preserve directory structure."""
        gitkeep = DATABASE_DIR / "labels" / ".gitkeep"
        assert gitkeep.exists(), f".gitkeep not found: {gitkeep}"


class TestDataDirectoryStructure:
    """Tests for data directory structure."""

    def test_database_directory_exists(self) -> None:
        """Test that database directory exists."""
        assert DATABASE_DIR.exists(), f"Database directory not found: {DATABASE_DIR}"

    def test_data_directory_exists(self) -> None:
        """Test that data directory exists."""
        data_dir = DATABASE_DIR / "data"
        assert data_dir.exists(), f"Data directory not found: {data_dir}"

    def test_video_directory_exists(self) -> None:
        """Test that video directory exists (may be empty before download)."""
        video_dir = DATABASE_DIR / "data" / "20bn-something-something-v2"
        assert video_dir.exists(), f"Video directory not found: {video_dir}"


class TestScriptsDirectory:
    """Tests for scripts directory structure."""

    def test_scripts_directory_exists(self) -> None:
        """Test that scripts directory exists."""
        assert SCRIPTS_DIR.exists(), f"Scripts directory not found: {SCRIPTS_DIR}"

    def test_setup_data_script_exists(self) -> None:
        """Test that setup_data.sh exists."""
        script = SCRIPTS_DIR / "setup_data.sh"
        assert script.exists(), f"setup_data.sh not found: {script}"
