import darshan
import platform
python_version = platform.python_version_tuple()

# shim for convenient Python 3.9 importlib.resources
# interface
if int(python_version[1]) < 9 and int(python_version[0]) == 3:
    import importlib_resources
else:
    # see: https://github.com/python/mypy/issues/1153
    import importlib.resources as importlib_resources # type: ignore


import pytest

try:
    import darshan_logs
    has_log_repo = True
except ImportError:
    has_log_repo = False

def pytest_configure():
    pytest.has_log_repo = has_log_repo

class CustomNode:
    def __init__(self, mimic_item):
        for attr in dir(mimic_item):
            if attr not in ["own_markers", "iter_markers"] and "__" not in attr:
                setattr(self, attr, getattr(mimic_item, attr))

def pytest_collection_modifyitems(config, items):
    new_items = []
    skip_indices = []
    for i, test_item in enumerate(items.copy()):
        param_marker = test_item.own_markers[0]
        num_args = int(len(param_marker.args) / 2)
        for arg_num in range(num_args):
            arg_name = param_marker.args[arg_num]
            arg_values = param_marker.args[arg_num + 1]
            for arg_val_num, arg_value in enumerate(arg_values[:]):
                skipper_found = 0
                for arg_sub_value in arg_value:
                    if isinstance(arg_sub_value, darshan.log_utils.Skipper):
                        err_msg = arg_sub_value.err_msg
                        if arg_val_num not in skip_indices:
                            skip_indices.append(arg_val_num)
                            customnode = CustomNode(test_item)
                            customnode.own_markers = pytest.mark.skip
                            def iter_marks(name):
                                yield pytest.mark.skip(reason=err_msg)
                            customnode.iter_markers = iter_marks
                            new_items.append(customnode)

    for i, item in enumerate(items.copy()):
        if i not in skip_indices:
            new_items.append(item)

    items.clear()
    items.extend(new_items)


@pytest.fixture
def log_repo_files():
    # provide a convenient way to access the list
    # of all *.darshan log files in the logs repo,
    # returning a list of absolute file paths to
    # the logs
    if pytest.has_log_repo:
        p = importlib_resources.files('darshan_logs')
        return [str(p) for p in p.glob('**/*.darshan')]

@pytest.fixture
def select_log_repo_file(log_repo_files, filename):
    # return the absolute path to a log repo
    # file based on its filename
    if pytest.has_log_repo:
        for path in log_repo_files:
            if filename in path:
                return path
