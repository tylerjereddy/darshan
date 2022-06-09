import numpy as np
import h5py
from mpi4py import MPI
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.mpi
def test_h5oopen_h5py_roundtrip(tmpdir):
    # regression test for gh-690
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # round trip HDF5 IO test
    with tmpdir.as_cwd():
        file_path = f"./test_rank_{rank}.hdf5"
        n_bytes = 10 * (rank + 1)
        bytes_to_write = np.ones(shape=n_bytes, dtype=np.int8)

        with h5py.File(file_path, "w") as f:
            f.create_dataset("dataset", data=bytes_to_write)

        with h5py.File(file_path, "r") as g:
            retrieved_data = np.asarray(g['dataset'])
            assert_array_equal(retrieved_data, bytes_to_write)
