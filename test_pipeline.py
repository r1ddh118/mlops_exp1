"""
test_pipeline.py

Unit tests for the two pipeline stages:
    - data_ingestion.py   (dvc stage: collection)
    - data_preprocessing.py (dvc stage: preprocessing)

These tests stub out the project's `logger` and `exception` modules
(so the suite can run even if those files aren't present / don't want
real logging side effects), then exercise the public functions of
both pipeline modules with pytest fixtures, tmp_path, and mocks -
no real Census file or params.yaml is required.

Run with:
    pytest test_pipeline.py -v
"""

import os
import sys
import types
import builtins
from unittest.mock import patch, mock_open

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub the project's `logger` and `exception` modules before importing the
# pipeline modules under test, since those two files aren't part of this
# test target and may not exist / may not be importable in isolation.
# ---------------------------------------------------------------------------
def _install_stub_dependencies():
    if "logger" not in sys.modules:
        logger_stub = types.ModuleType("logger")

        class _DummyLogger:
            def info(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        def get_logger(name):
            return _DummyLogger()

        logger_stub.get_logger = get_logger
        sys.modules["logger"] = logger_stub

    if "exception" not in sys.modules:
        exception_stub = types.ModuleType("exception")

        class PipelineException(Exception):
            def __init__(self, error, error_detail=None):
                super().__init__(str(error))
                self.original_error = error

        exception_stub.PipelineException = PipelineException
        sys.modules["exception"] = exception_stub


_install_stub_dependencies()

from src import data_ingestion  # noqa: E402
from src import data_preprocessing  # noqa: E402

from exception import PipelineException  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_raw_df():
    """A tiny stand-in for a raw survey sheet read from Excel."""
    return pd.DataFrame(
        {
            "PRICE": [150000, 9, 220000, 999999],
            "SQFT": [1200, 1400, 9, 1600],
            "BEDROOMS": [3, 2, 4, 3],
            "JFLAG1": [0, 1, 0, 1],
            "jflag2": [1, 0, 0, 0],
            "WEIGHT": [1.2, 1.1, 0.9, 1.0],
            "WGTADJ": [1.0, 1.0, 1.0, 1.0],
            "CONTROL": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def ingestion_params():
    return {
        "raw_data_path": "data/raw/puf2022.xls",
        "sheet_name": "puf2022",
        "ingested_data_path": "dataset/housing.csv",
    }


@pytest.fixture
def preprocessing_params():
    return {
        "input_path": "dataset/housing.csv",
        "output_path": "dataset/preprocessed.csv",
        "price_min_valid": 10,
        "sqft_min_valid": 10,
    }


# ===========================================================================
# data_ingestion.py
# ===========================================================================

class TestLoadParamsIngestion:
    def test_load_params_returns_ingestion_section(self, ingestion_params):
        full_params_yaml = "data_ingestion:\n  raw_data_path: data/raw/puf2022.xls\n"
        with patch("builtins.open", mock_open(read_data=full_params_yaml)):
            with patch(
                "yaml.safe_load",
                return_value={"data_ingestion": ingestion_params},
            ):
                result = data_ingestion.load_params("params.yaml")
        assert result == ingestion_params

    def test_load_params_missing_file_raises_pipeline_exception(self):
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            with pytest.raises(PipelineException):
                data_ingestion.load_params("does_not_exist.yaml")

    def test_load_params_missing_key_raises_pipeline_exception(self):
        # params.yaml exists but has no "data_ingestion" section
        with patch("builtins.open", mock_open(read_data="other_stage: {}")):
            with patch("yaml.safe_load", return_value={"other_stage": {}}):
                with pytest.raises(PipelineException):
                    data_ingestion.load_params("params.yaml")


class TestIngestData:
    def test_ingest_data_reads_excel_and_writes_csv(self, tmp_path, sample_raw_df):
        raw_path = tmp_path / "puf2022.xls"
        raw_path.write_text("dummy")  # just needs to exist
        out_path = tmp_path / "out" / "housing.csv"

        with patch("pandas.read_excel", return_value=sample_raw_df) as mock_read:
            result_df = data_ingestion.ingest_data(
                raw_data_path=str(raw_path),
                sheet_name="puf2022",
                ingested_data_path=str(out_path),
            )

        mock_read.assert_called_once_with(str(raw_path), sheet_name="puf2022")
        assert out_path.exists()
        assert result_df.shape == sample_raw_df.shape

        reloaded = pd.read_csv(out_path)
        assert list(reloaded.columns) == list(sample_raw_df.columns)
        assert len(reloaded) == len(sample_raw_df)

    def test_ingest_data_creates_output_directory(self, tmp_path, sample_raw_df):
        raw_path = tmp_path / "puf2022.xls"
        raw_path.write_text("dummy")
        nested_out_path = tmp_path / "a" / "b" / "c" / "housing.csv"

        with patch("pandas.read_excel", return_value=sample_raw_df):
            data_ingestion.ingest_data(
                raw_data_path=str(raw_path),
                sheet_name="puf2022",
                ingested_data_path=str(nested_out_path),
            )

        assert nested_out_path.parent.is_dir()
        assert nested_out_path.exists()

    def test_ingest_data_missing_raw_file_raises_pipeline_exception(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.xls"
        with pytest.raises(PipelineException):
            data_ingestion.ingest_data(
                raw_data_path=str(missing_path),
                sheet_name="puf2022",
                ingested_data_path=str(tmp_path / "housing.csv"),
            )

    def test_ingest_data_wraps_read_excel_errors(self, tmp_path):
        raw_path = tmp_path / "puf2022.xls"
        raw_path.write_text("dummy")

        with patch("pandas.read_excel", side_effect=ValueError("bad sheet")):
            with pytest.raises(PipelineException):
                data_ingestion.ingest_data(
                    raw_data_path=str(raw_path),
                    sheet_name="nonexistent_sheet",
                    ingested_data_path=str(tmp_path / "housing.csv"),
                )


class TestIngestionMain:
    def test_main_calls_ingest_data_with_loaded_params(self, ingestion_params):
        with patch.object(data_ingestion, "load_params", return_value=ingestion_params) as mock_load:
            with patch.object(data_ingestion, "ingest_data") as mock_ingest:
                data_ingestion.main()

        mock_load.assert_called_once()
        mock_ingest.assert_called_once_with(
            raw_data_path=ingestion_params["raw_data_path"],
            sheet_name=ingestion_params["sheet_name"],
            ingested_data_path=ingestion_params["ingested_data_path"],
        )

    def test_main_propagates_exceptions(self):
        with patch.object(data_ingestion, "load_params", side_effect=PipelineException("boom")):
            with pytest.raises(PipelineException):
                data_ingestion.main()


# ===========================================================================
# data_preprocessing.py
# ===========================================================================

class TestLoadParamsPreprocessing:
    def test_load_params_returns_preprocessing_section(self, preprocessing_params):
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch(
                "yaml.safe_load",
                return_value={"data_preprocessing": preprocessing_params},
            ):
                result = data_preprocessing.load_params("params.yaml")
        assert result == preprocessing_params

    def test_load_params_missing_file_raises_pipeline_exception(self):
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            with pytest.raises(PipelineException):
                data_preprocessing.load_params("does_not_exist.yaml")


class TestDropIrrelevantColumns:
    def test_drops_j_flag_columns_case_insensitively(self, sample_raw_df):
        result = data_preprocessing.drop_irrelevant_columns(sample_raw_df)
        assert "JFLAG1" not in result.columns
        assert "jflag2" not in result.columns

    def test_drops_survey_design_columns(self, sample_raw_df):
        result = data_preprocessing.drop_irrelevant_columns(sample_raw_df)
        for col in ["WEIGHT", "WGTADJ", "CONTROL"]:
            assert col not in result.columns

    def test_keeps_non_j_prefixed_predictive_columns(self, sample_raw_df):
        result = data_preprocessing.drop_irrelevant_columns(sample_raw_df)
        for col in ["PRICE", "SQFT", "BEDROOMS"]:
            assert col in result.columns

    def test_missing_design_columns_do_not_raise(self):
        df = pd.DataFrame({"PRICE": [100000], "SQFT": [1000]})
        result = data_preprocessing.drop_irrelevant_columns(df)
        assert list(result.columns) == ["PRICE", "SQFT"]

    def test_no_j_columns_present(self):
        df = pd.DataFrame({"PRICE": [100000], "WEIGHT": [1.0]})
        result = data_preprocessing.drop_irrelevant_columns(df)
        assert "PRICE" in result.columns
        assert "WEIGHT" not in result.columns


class TestRemoveInvalidRows:
    def test_removes_sentinel_price_and_sqft_rows(self):
        df = pd.DataFrame(
            {
                "PRICE": [150000, 9, 220000, 999999],
                "SQFT": [1200, 1400, 9, 1600],
            }
        )
        result = data_preprocessing.remove_invalid_rows(df, price_min_valid=10, sqft_min_valid=10)
        # row idx 1 (PRICE=9) and idx 2 (SQFT=9) should be dropped
        assert len(result) == 2
        assert 9 not in result["PRICE"].values
        assert 9 not in result["SQFT"].values

    def test_resets_index(self):
        df = pd.DataFrame({"PRICE": [9, 150000, 220000], "SQFT": [1200, 1300, 1400]})
        result = data_preprocessing.remove_invalid_rows(df, price_min_valid=10, sqft_min_valid=10)
        assert list(result.index) == list(range(len(result)))

    def test_no_rows_removed_when_all_valid(self):
        df = pd.DataFrame({"PRICE": [150000, 220000], "SQFT": [1200, 1400]})
        result = data_preprocessing.remove_invalid_rows(df, price_min_valid=10, sqft_min_valid=10)
        assert len(result) == 2

    def test_all_rows_removed_when_all_invalid(self):
        df = pd.DataFrame({"PRICE": [1, 2], "SQFT": [1, 2]})
        result = data_preprocessing.remove_invalid_rows(df, price_min_valid=10, sqft_min_valid=10)
        assert len(result) == 0


class TestPreprocessData:
    def test_full_pipeline_writes_expected_csv(self, tmp_path, sample_raw_df):
        input_path = tmp_path / "housing.csv"
        sample_raw_df.to_csv(input_path, index=False)
        output_path = tmp_path / "out" / "preprocessed.csv"

        result_df = data_preprocessing.preprocess_data(
            input_path=str(input_path),
            output_path=str(output_path),
            price_min_valid=10,
            sqft_min_valid=10,
        )

        assert output_path.exists()
        # j-flag and design columns should be gone
        for col in ["JFLAG1", "jflag2", "WEIGHT", "WGTADJ", "CONTROL"]:
            assert col not in result_df.columns
        # sentinel rows (PRICE=9 / SQFT=9) should be gone
        assert (result_df["PRICE"] >= 10).all()
        assert (result_df["SQFT"] >= 10).all()

        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == len(result_df)

    def test_drops_duplicate_rows(self, tmp_path):
        df = pd.DataFrame(
            {
                "PRICE": [150000, 150000, 220000],
                "SQFT": [1200, 1200, 1400],
            }
        )
        input_path = tmp_path / "housing.csv"
        df.to_csv(input_path, index=False)
        output_path = tmp_path / "preprocessed.csv"

        result_df = data_preprocessing.preprocess_data(
            input_path=str(input_path),
            output_path=str(output_path),
            price_min_valid=10,
            sqft_min_valid=10,
        )

        assert len(result_df) == 2  # duplicate collapsed

    def test_missing_input_file_raises_pipeline_exception(self, tmp_path):
        with pytest.raises(PipelineException):
            data_preprocessing.preprocess_data(
                input_path=str(tmp_path / "nope.csv"),
                output_path=str(tmp_path / "out.csv"),
                price_min_valid=10,
                sqft_min_valid=10,
            )

    def test_creates_output_directory(self, tmp_path, sample_raw_df):
        input_path = tmp_path / "housing.csv"
        sample_raw_df.to_csv(input_path, index=False)
        nested_output = tmp_path / "x" / "y" / "z" / "preprocessed.csv"

        data_preprocessing.preprocess_data(
            input_path=str(input_path),
            output_path=str(nested_output),
            price_min_valid=10,
            sqft_min_valid=10,
        )

        assert nested_output.exists()


class TestPreprocessingMain:
    def test_main_calls_preprocess_data_with_loaded_params(self, preprocessing_params):
        with patch.object(data_preprocessing, "load_params", return_value=preprocessing_params) as mock_load:
            with patch.object(data_preprocessing, "preprocess_data") as mock_preprocess:
                data_preprocessing.main()

        mock_load.assert_called_once()
        mock_preprocess.assert_called_once_with(
            input_path=preprocessing_params["input_path"],
            output_path=preprocessing_params["output_path"],
            price_min_valid=preprocessing_params["price_min_valid"],
            sqft_min_valid=preprocessing_params["sqft_min_valid"],
        )

    def test_main_propagates_exceptions(self):
        with patch.object(data_preprocessing, "load_params", side_effect=PipelineException("boom")):
            with pytest.raises(PipelineException):
                data_preprocessing.main()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
