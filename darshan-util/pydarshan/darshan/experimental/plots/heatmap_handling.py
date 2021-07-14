"""
Module of data pre-processing functions for constructing the heatmap figure.
"""
from typing import Dict, Any, Tuple, Sequence, TypedDict

import pandas as pd
import numpy as np


class SegDict(TypedDict):
    """
    Custom type hint class for `dict_list` argument in `get_rd_wr_dfs()`.
    """

    id: int
    rank: int
    hostname: str
    write_count: int
    read_count: int
    write_segments: pd.DataFrame
    read_segments: pd.DataFrame


def get_rd_wr_dfs(dict_list: Sequence[SegDict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses the DXT records to construct individual
    dataframes for both read and write segments.

    Parameters
    ----------

    dict_list: a list of DXT records, where each record is a
    Python dictionary with the following keys: 'id', 'rank',
    'hostname', 'write_count', 'read_count', 'write_segments',
    and 'read_segments'. The read/write data is stored in
    ``read_segments`` and ``write_segments``, where each is a
    ```pd.DataFrame`` containing the following data (columns):
    'offset', 'length', 'start_time', 'end_time'.

    Returns
    -------

    Tuple of form ``(read_df, write_df)``,
    where each tuple element is a ``pd.DataFrame`` object
    containing all of the read and write events.

    Notes
    -----

    Used in ``get_single_df_dict()``.

    Examples
    --------

    ``dict_list`` and ``(read_df, write_df)``
    generated from ``tests/input/sample-dxt-simple.darshan``:

        dict_list = [
            {
                'id': 14388265063268455899,
                'rank': 0,
                'hostname': 'sn176.localdomain',
                'write_count': 1,
                'read_count': 0,
                'write_segments':
                    offset  length  start_time  end_time
                    0       0      40    0.103379  0.103388,
                'read_segments':
                    Empty DataFrame
                    Columns: []
                    Index: []
            },
            {
                'id': 9457796068806373448,
                'rank': 0,
                'hostname': 'sn176.localdomain',
                'write_count': 1,
                'read_count': 0,
                'write_segments':
                    offset  length  start_time  end_time
                    0       0    4000    0.104217  0.104231,
                'read_segments':
                    Empty DataFrame
                    Columns: []
                    Index: []
            },
        ]

        (read_df, write_df) = (
            Empty DataFrame
            Columns: []
            Index: [],
            length  start_time  end_time  rank
            0      40    0.103379  0.103388     0
            1    4000    0.104217  0.104231     0
        )

    """
    # columns to drop when accumulating the dataframes.
    # Currently "offset" data is not utilized
    drop_columns = ["offset"]
    # create empty arrays to store
    # read/write segment dataframes
    read_df_list = []
    write_df_list = []
    # iterate over all records/dictionaries
    # to pull out the dataframes
    for _dict in dict_list:
        # collect the read and write segment dataframes
        rd_seg_df = _dict["read_segments"]
        wr_seg_df = _dict["write_segments"]

        if rd_seg_df.size:
            # drop unused columns from the dataframe
            rd_seg_df = rd_seg_df.drop(columns=drop_columns)
            # create new column for the ranks
            rd_seg_df["rank"] = _dict["rank"]
            # add the dataframe to the list
            read_df_list.append(rd_seg_df)

        if wr_seg_df.size:
            # drop unused columns from the dataframe
            wr_seg_df = wr_seg_df.drop(columns=drop_columns)
            # create new column for the ranks
            wr_seg_df["rank"] = _dict["rank"]
            # add the dataframe to the list
            write_df_list.append(wr_seg_df)

    if read_df_list:
        # concatenate the list of pandas dataframes into
        # a single one with new row indices
        read_df = pd.concat(read_df_list, ignore_index=True)
    else:
        # if the list is empty assign an empty dataframe
        read_df = pd.DataFrame()

    if write_df_list:
        # concatenate the list of pandas dataframes into
        # a single one with new row indices
        write_df = pd.concat(write_df_list, ignore_index=True)
    else:
        # if the list is empty assign an empty dataframe
        write_df = pd.DataFrame()

    return read_df, write_df


def get_single_df_dict(
    report: Any,
    mods: Sequence[str] = ["DXT_POSIX"],
    ops: Sequence[str] = ["read", "write"],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Reorganizes segmented read/write data into a single ``pd.DataFrame``
    and stores them in a dictionary with an entry for each DXT module.

    Parameters
    ----------

    report: a ``darshan.DarshanReport``.

    mods: a list of keys designating which Darshan modules to use for
    data aggregation. Default is ``["DXT_POSIX"]``.

    ops: a list of keys designating which Darshan operations to use for
    data aggregation. Default is ``["read", "write"]``.

    Returns
    -------

    flat_data_dict: a nested dictionary where the input module
    keys (i.e. "DXT_POSIX") are the top level keys, which contain
    an entry for each input operation (i.e. "read"/"write") that
    map to dataframes containing all events for the specified operation.

    Examples
    --------
    `flat_data_dict` generated from `tests/input/sample-dxt-simple.darshan`:
        {
            'DXT_POSIX':
                {
                    'read':
                        Empty DataFrame
                        Columns: []
                        Index: [],
                    'write':
                        length  start_time  end_time  rank
                        0      40    0.103379  0.103388     0
                        1    4000    0.104217  0.104231     0
                }
        }

    """
    # initialize an empty dictionary for storing
    # module and read/write data
    flat_data_dict = {}  # type: Dict[str, Dict[str, pd.DataFrame]]
    # iterate over the modules (i.e. DXT_POSIX)
    for module_key in mods:
        # read in the module data, update the name records
        report.mod_read_all_dxt_records(module_key, dtype="pandas")
        # retrieve the list of records in pd.DataFrame() form
        dict_list = report.records[module_key].to_df()
        # retrieve the list of read/write dataframes from the list of records
        read_df, write_df = get_rd_wr_dfs(dict_list=dict_list)
        # create empty dictionary for each module
        flat_data_dict[module_key] = {}
        if "read" in ops:
            # add the concatenated dataframe to the flat dictionary
            flat_data_dict[module_key]["read"] = read_df
        if "write" in ops:
            # add the concatenated dataframe to the flat dictionary
            flat_data_dict[module_key]["write"] = write_df

    return flat_data_dict


def get_aggregate_data(
    report: Any,
    mods: Sequence[str] = ["DXT_POSIX"],
    ops: Sequence[str] = ["read", "write"],
) -> pd.DataFrame:
    """
    Aggregates the data based on which
    modules and operations are selected.

    Parameters
    ----------

    report: a ``darshan.DarshanReport``.

    mods: a list of keys designating which Darshan modules to use for
    data aggregation. Default is ``["DXT_POSIX"]``.

    ops: a list of keys designating which Darshan operations to use for
    data aggregation. Default is ``["read", "write"]``.

    Returns
    -------

    agg_df: a ``pd.DataFrame`` containing the aggregated data determined
    by the input modules and operations.

    Raises
    ------

    ValueError: raised if the selected modules/operations
    don't contain any data.

    Notes
    -----
    Since read and write events are considered unique events, if both are
    selected their dataframes are simply concatenated.

    Examples
    --------
    `agg_df` generated from `tests/input/sample-dxt-simple.darshan`:

            length  start_time  end_time  rank
        0      40    0.103379  0.103388     0
        1    4000    0.104217  0.104231     0

    """
    # collect the concatenated dataframe data from the darshan report
    df_dict = get_single_df_dict(report=report, mods=mods, ops=ops)
    # TODO: generalize for all DXT modules, for now manually set `DXT_POSIX`
    module_key = "DXT_POSIX"
    # iterate over each dataframe based on which operations are selected
    df_list = []
    for op_key, op_df in df_dict[module_key].items():
        # if the dataframe has data, append it to the list
        if op_df.size:
            df_list.append(op_df)

    if df_list:
        # if there are dataframes in the list, concatenate them into 1 dataframe
        agg_df = pd.concat(df_list, ignore_index=True)
    else:
        raise ValueError(f"No data available for selected module(s) and operation(s).")

    return agg_df


def calc_prop_data_sum(
    tmin: float,
    tmax: float,
    total_elapsed: np.ndarray,
    total_data: np.ndarray,
) -> float:
    """
    Calculates the proportion of data read/written in the
    time interval of a single bin in ``get_heatmap_data``.

    Parameters
    ----------

    tmin: the lower bound of the time interval for a given bin.

    tmax: the upper bound of the time interval for a given bin.

    total_elapsed: an array of the elapsed times for every event
    that occurred within the time interval of a given bin.

    total_data: an array of the data totals for every event that
    occurred within the time interval of a given bin.

    Returns
    -------

    prop_data_sum: the amount of data read/written in the time
    interval of a given bin.

    """
    # calculate the elapsed time
    partial_elapsed = tmax - tmin
    # calculate the ratio of the elapsed time
    # to the total read/write event time
    proportionate_time = partial_elapsed / total_elapsed
    # calculate the amount of data read/written in the elapsed
    # time (assuming a constant read/write rate)
    proportionate_data = proportionate_time * total_data
    # sum the data
    prop_data_sum = proportionate_data.sum()
    return prop_data_sum


def get_heatmap_data(agg_df: pd.DataFrame, xbins: int) -> np.ndarray:
    """
    Builds an array similar to a 2D-histogram, where the y data is the unique
    ranks and the x data is time. Each bin is populated with the data sum
    and/or proportionate data sum for all IO events read/written during the
    time spanned by the bin.

    Parameters
    ----------

    agg_df: a ``pd.DataFrame`` containing the aggregated data determined
    by the input modules and operations.

    xbins: the number of x-axis bins to create.

    Returns
    -------

    hmap_data: ``NxM`` array, where ``N`` is the number of unique ranks
    and ``M`` is the number of x-axis bins. Each element contains the
    data read/written by the corresponding rank in the x-axis bin time
    interval.

    """
    # get the unique ranks
    unique_ranks = np.unique(agg_df["rank"].values)

    # generate the bin edges by generating an array of length n_bins+1, then
    # taking pairs of data points as the min/max bin value
    min_time = 0.0
    max_time = agg_df["end_time"].values.max()
    bin_edge_data = np.linspace(min_time, max_time, xbins + 1)
    cats_start = pd.get_dummies(pd.cut(agg_df["start_time"], bin_edge_data))
    cats_start["rank"] = agg_df["rank"]
    cats_start["length"] = agg_df["length"]
    cats_end = pd.get_dummies(pd.cut(agg_df["end_time"], bin_edge_data))
    cats_end["rank"] = agg_df["rank"]
    cats_end["length"] = agg_df["length"]
    cats = cats_start.where(cats_start > cats_end, cats_end)
    cats.iloc[:, :xbins] = cats.iloc[:, :xbins].replace(0, np.nan)
    print("combined cats:\n", cats)
    # at this point we have a dataframe of dummy (indicator)
    # variables for each of the time segments (bins) for
    # each IO event (row, which corresponds to a rank)
    # each time bin that has an event has a 1 in it, otherwise
    # NaN

    # now, for each row (IO event) we want to fill in any empty (NaN)
    # bins between filled bins because those are time spans b/w start
    # and stop events; divide the indicators by the number of total
    # bins spanned by the event that is filled in

    # this is actually an inner linear interpolation, followed by
    # division to produce weights on the bin events
    cats.iloc[:, :xbins] = cats.iloc[:, :xbins].interpolate(method='linear',
                                                            limit_area='inside',
                                                            axis=1)
    print("combined cats interpolated:\n", cats)
    sums = cats.iloc[:, :xbins].sum(axis=1) 
    print("sums:", sums)
    cats.iloc[:, :xbins] = cats.iloc[:, :xbins].div(sums, axis=0)

    # each full or fractional bin event is now multiplied by
    # the bytes data
    cats.iloc[:, :xbins] = cats.iloc[:, :xbins].mul(cats['length'], axis=0)
    hmap_data = cats
    hmap_data.set_index('rank', inplace=True)
    hmap_data.drop('length', inplace=True, axis=1)
    print("*** final hmap_data:\n", hmap_data)
    grouped_res = hmap_data.groupby('rank').sum()
    print("grouped result:", grouped_res)
    array_res = grouped_res.to_numpy()
    print("grouped ARRAY result:", array_res)
    return array_res
