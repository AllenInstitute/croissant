from typing import Tuple
from pathlib import Path

import pytest
import h5py
import numpy as np


@pytest.fixture()
def trace_file_fixture(tmp_path: Path, request) -> Tuple[Path, dict]:
    """Fixture that allows parametrized optical physiology trace files
    (*.h5) to be generated."""

    trace_name = request.param.get("trace_filename", "mock_trace.h5")

    trace_data = request.param.get("trace_data",
                                   np.arange(100).reshape((5, 20)))
    trace_data_key = request.param.get("trace_data_key", "data")

    trace_names = request.param.get("trace_names", ['0', '4', '1', '3', '2'])
    trace_names_key = request.param.get("trace_names_key", "roi_names")

    fixture_params = {"trace_data": trace_data,
                      "trace_data_key": trace_data_key,
                      "trace_names": trace_names,
                      "trace_names_key": trace_names_key}

    trace_path = tmp_path / trace_name
    with h5py.File(trace_path, "w") as f:
        f[trace_data_key] = trace_data
        formatted_trace_names = np.array(trace_names).astype(np.string_)
        f.create_dataset(trace_names_key, data=formatted_trace_names)

    return trace_path, fixture_params
