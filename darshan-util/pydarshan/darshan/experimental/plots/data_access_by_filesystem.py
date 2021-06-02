"""
Draft code for the `data access by category` section
of Phil's hand drawing of future report layout.
"""

import os
import pathlib
import collections

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import darshan

pd.options.display.max_colwidth = 100

def produce_dxt_counter_df(report, module):
    # accepts a darshan report object and string
    # with the name of the module
    # the DXT_POSIX and DXT_MPIIO modules
    # accepted by this function require special
    # handling to produce a convenient dataframe
    if module not in {'DXT_POSIX', 'DXT_MPIIO'}:
        raise ValueError(f"module {module} not supported by produce_dxt_counter_df")

    # it seems that the DXT data is initially provided as a
    # dictionary, even if requesting via to_df(), so
    # a bit of processing to do to be able to return
    # a dataframe that can be handled similarly to i.e.,
    # the POSIX module
    dxt_data_dict = report.records[module].to_df()[0]
    # now, DXT tracing organizes first by file ID, then by rank number

    # I believe that to provide consistency with the POSIX-style
    # dataframe interface in my current workflow, one approach
    # might be to have a single dataframe row per file ID
    # along with the "length" (in bytes) columns summed to a
    # single value per file for the read_segments and write_segments
    # dictionary fields (which are themselves dataframes)

    # NOTE: the DXT log file I'm currently working with is likely too
    # simple to handle the general/more complex cases
    # TODO: try with DXT-enabled logs that trace many different files
    # and get the summarized dataframe data produced correctly

    # for the moment, the dictionaries/DXT data structures
    # I have access to are fairly flat
    id_field = dxt_data_dict['id']
    read_segments_df = dxt_data_dict['read_segments']
    write_segments_df = dxt_data_dict['write_segments']

    # compound to {module}_BYTES_READ and {module}_BYTES_WRITTEN dataframe fields
    # for similarity with i.e., POSIX mod df layout
    if read_segments_df.empty:
        bytes_read_field = 0
    else:
        bytes_read_field = read_segments_df['length'].sum()

    if write_segments_df.empty:
        bytes_write_field = 0
    else:
        bytes_write_field = write_segments_df['length'].sum()

    # produce a DXT "equivalent" of rec_counters df
    rec_counters = pd.DataFrame({'id': id_field,
                                f'{module}_BYTES_READ': bytes_read_field,
                                f'{module}_BYTES_WRITTEN': bytes_write_field,
                                },
                                index=[0])
    return rec_counters


def convert_file_path_to_root_path(file_path):
    path_parts = pathlib.Path(file_path).parts
    filesystem_root = ''.join(path_parts[:2])
    return filesystem_root

def convert_file_id_to_path(input_id, file_id_dict):
    result_found = False
    for file_id_hash, file_path in file_id_dict.items():
        if np.allclose(input_id, file_id_hash):
            result_found = True
            return file_path
    if not result_found:
        msg = f'could not find path for file ID: {input_id}'
        raise ValueError(msg)

def identify_filesystems(file_id_dict, verbose=False):
    # file_id_dict is from report.data["name_records"]

    # the function returns a list of unique filesystems
    # (path roots)

    filesystem_roots = []
    excluded = ['<STDIN>', '<STDOUT>', '<STDERR>']
    for file_id_hash, file_path in file_id_dict.items():
        filesystem_root = convert_file_path_to_root_path(file_path=file_path)
        if filesystem_root not in filesystem_roots:
            if filesystem_root not in excluded:
                filesystem_roots.append(filesystem_root)
    if verbose:
        print("filesystem_roots:", filesystem_roots)
    return filesystem_roots

def per_filesystem_unique_file_read_write_counter(report, filesystem_roots, verbose=False):
    # we are interested in finding all unique files that we have read
    # at least 1 byte from, or written at least 1 byte to
    # and then summing those counts per filesystem

    # report is a darshan.DarshanReport()
    # filesystem_roots is a list of unique filesystem root paths
    # from identify_filesystems()

    # returns: tuple
    # (read_groups, write_groups)
    # where each element of the tuple is a pandas
    # Series object with a format like the one shown below

    # filesystem_root
    # /tmp       1
    # /yellow    1

    # the int64 values in the Series are the counts
    # of unique files to which a single byte has been read
    # (or written) on a given filesystem (index)

    data_dict = {}

    for mod in report._modules.keys():
        if mod == 'LUSTRE':
            continue
        print("parsing mod:", mod)
        report.mod_read_all_records(mod, dtype='pandas')
        try:
            rec_counters = report.records[mod][0]['counters']
        except KeyError:
            # for DXT-related modules for example
            # we'll need separate handling to produce
            # a rec_counters dataframe
            rec_counters = produce_dxt_counter_df(report=report, module=mod)
        
        if mod == 'MPI-IO':
            subkey = mod.replace('-', '')
        else:
            subkey = mod
        # first, filter to produce a dataframe where (mod)_BYTES_READ >= 1
        # for each row (tracked event for a given rank or group of ranks)
        df_reads = rec_counters.loc[rec_counters[f'{subkey}_BYTES_READ'] >= 1]

        # similar filter for writing
        df_writes = rec_counters.loc[rec_counters[f'{subkey}_BYTES_WRITTEN'] >= 1]

        # add column with filepaths for each event
        df_reads = df_reads.assign(filepath=df_reads['id'].map(lambda a: convert_file_id_to_path(a, file_id_dict)))
        df_writes = df_writes.assign(filepath=df_writes['id'].map(lambda a: convert_file_id_to_path(a, file_id_dict)))

        # add column with filesystem root paths for each event
        df_reads = df_reads.assign(filesystem_root=df_reads['filepath'].map(lambda path: convert_file_path_to_root_path(path)))
        df_writes = df_writes.assign(filesystem_root=df_writes['filepath'].map(lambda path: convert_file_path_to_root_path(path)))

        # we're going to be combining data from different instrumentation
        # modules, so try to keep the data frames that are candidates for
        # aggregation as simple as possible
        # we should only need these fields:
        # filesystem_root
        # filepath
        # subkey_BYTES_READ OR subkey_BYTES_WRITTEN
        df_reads = df_reads[["filesystem_root", "filepath", f"{subkey}_BYTES_READ"]]
        df_writes = df_writes[["filesystem_root", "filepath", f"{subkey}_BYTES_WRITTEN"]]

        # groupby filesystem root and filepath, then get sizes
        # these are Pandas series objects where the index
        # is the name of the filesystem_root and the value
        # is the number of unique files counted per above criteria
        #read_groups = df_reads.groupby('filesystem_root')['filepath'].nunique()
        #write_groups = df_writes.groupby('filesystem_root')['filepath'].nunique()

        # if either of the dataframes are effectively empty we want
        # to produce a new dataframe with the filesystem_root values
        # and count values of 0 (for plotting purposes, etc.)
        if df_reads.empty:
            for idx, filesystem_root in enumerate(filesystem_roots):
                df_reads.loc[idx, "filesystem_root"] = filesystem_root
                df_reads.loc[idx, f"{subkey}_BYTES_READ"] = 0

        if df_writes.empty:
            for idx, filesystem_root in enumerate(filesystem_roots):
                df_writes.loc[idx, "filesystem_root"] = filesystem_root
                df_writes.loc[idx, f"{subkey}_BYTES_WRITTEN"] = 0

        data_dict[mod] = (df_reads, df_writes)
    # TODO: this data_dict also contains the bytes read/write
    # data needed for the other plots Phil asked for in the
    # "Data access by category" section of the report
    # (# bytes read & # bytes written), so should be
    # able to use a similar approach for aggregating
    # that data...

    # proceed with combining the data from the different
    # darshan instrumentation modules into single read
    # and write dataframes

    # the combined dataframe might have columns as follows:
#      filesystem_root filepath  POSIX_BYTES_READ
    combined_df_read = pd.DataFrame()
    combined_df_write = pd.DataFrame()
    for mod_name, mod_data in data_dict.items():
        # NOTE: performance on looping/appending like this
        # will probably be bad
        df_read = mod_data[0]
        df_write = mod_data[1]
        combined_df_read = combined_df_read.append(df_read)
        combined_df_write = combined_df_write.append(df_write)

#combined_df_read:
#      filesystem_root filepath  POSIX_BYTES_READ  MPIIO_BYTES_READ  DXT_POSIX_BYTES_READ  DXT_MPIIO_BYTES_READ
#      0         /yellow      NaN               0.0               NaN                   NaN                   NaN
#      1            /tmp      NaN               0.0               NaN                   NaN                   NaN
#      0         /yellow      NaN               NaN               0.0                   NaN                   NaN
#      1            /tmp      NaN               NaN               0.0                   NaN                   NaN
#      0         /yellow      NaN               NaN               NaN                   0.0                   NaN
#      1            /tmp      NaN               NaN               NaN                   0.0                   NaN
#      0         /yellow      NaN               NaN               NaN                   NaN                   0.0
#      1            /tmp      NaN               NaN               NaN                   NaN                   0.0

#combined_df_write:
   #filesystem_root                                                                  filepath  POSIX_BYTES_WRITTEN  MPIIO_BYTES_WRITTEN  DXT_POSIX_BYTES_WRITTEN  DXT_MPIIO_BYTES_WRITTEN
#0         /yellow  /yellow/usr/projects/eap/users/treddy/simple_dxt_mpi_io_darshan/test.out               4000.0                  NaN                      NaN                      NaN
#0            /tmp                   /tmp/ompi.sn176.28751/jf.29186/1/test.out_cid-0-3400.sm                 40.0                  NaN                      NaN                      NaN
#0         /yellow  /yellow/usr/projects/eap/users/treddy/simple_dxt_mpi_io_darshan/test.out                  NaN               4000.0                      NaN                      NaN
#0            /tmp                   /tmp/ompi.sn176.28751/jf.29186/1/test.out_cid-0-3400.sm                  NaN                  NaN                     40.0                      NaN
#0         /yellow  /yellow/usr/projects/eap/users/treddy/simple_dxt_mpi_io_darshan/test.out                  NaN                  NaN                      NaN                   4000.0



    # next, we want to produce a dataframe with this column format:
    # filesystem_root | count_unique_files_read_from | count_unique_files_written_to
    col_list = list(combined_df_read)[2:]
    combined_df_read['BYTES_READ'] = combined_df_read[col_list].sum(axis=1)
    combined_df_read = combined_df_read[['filesystem_root', 'filepath', 'BYTES_READ' ]].drop_duplicates()

    col_list = list(combined_df_write)[2:]
    combined_df_write['BYTES_WRITTEN'] = combined_df_write[col_list].sum(axis=1)
    combined_df_write = combined_df_write[['filesystem_root', 'filepath', 'BYTES_WRITTEN' ]].drop_duplicates()

    #combined_df_read = combined_df_read.groupby('filesystem_root')
    #print("grouped:\n")
    #combined_df_read.apply(print)
    #combined_df_write = combined_df_write.groupby('filesystem_root')
    #print("grouped:\n")
    #combined_df_write.apply(print)

    if verbose:
        print('-' * 10)
        print("combined_df_read:\n", combined_df_read)
        print("combined_df_write:\n", combined_df_write)
        print('-' * 10)
    return (read_groups, write_groups)

def plot_series_files_rw(file_rd_series, file_wr_series, ax, log_filename):
    # plot the number of unique files per filesystem to which a single
    # byte has been read or written

    # file_rd_series and file_wr_series are pandas
    # Series objects with filesystems for indices
    # and int64 counts of unique files for values

    # ax is a matplotlib axis object

    df = pd.concat([file_rd_series, file_wr_series], axis=1)
    df.columns = ['read', 'write']
    #file_rd_series.plot(ax=ax, kind='barh', xlabel=None, ylabel=None, color='red', alpha=0.5, width=0.1)
    #file_wr_series.plot(ax=ax, kind='barh', xlabel=None, ylabel=None, color='blue', alpha=0.5, width=0.1)
    width = 0.1
    df.plot(ax=ax, kind='barh', xlabel=None, ylabel=None, alpha=0.5, width=width)
    print("df:", df)
    # put values next to bars
    [ax.text(v, i - width, '{:.0f}'.format(v)) for i, v in enumerate(file_rd_series)]
    [ax.text(v, i, '{:.0f}'.format(v)) for i, v in enumerate(file_wr_series)]
    ax.set_xlabel('# unique files')
    ax.set_ylabel('')
    ax.legend(['read', 'write'])

if __name__ == '__main__':
    # produce sample plots for some of the logs
    # available in our test suite
    root_path = 'tests/input'
    log_files = ['sample-dxt-simple.darshan', 'sample.darshan', 'sample-goodost.darshan']
    for idx, log_file in enumerate(log_files):
        fig = plt.figure()
        fig.suptitle(f"Data Access by Category for log file: '{log_file}'")
        ax_bytes = fig.add_subplot(1, 2, 1)
        ax_files = fig.add_subplot(1, 2, 2)
        log_path = os.path.join(root_path, log_file)
        filename = os.path.basename(log_path)
        report = darshan.DarshanReport(log_path, read_all=True)
        file_id_dict = report.data["name_records"]
        filesystem_roots = identify_filesystems(file_id_dict=file_id_dict, verbose=True)
        file_rd_series, file_wr_series = per_filesystem_unique_file_read_write_counter(report=report, filesystem_roots=filesystem_roots, verbose=True)
        plot_series_files_rw(file_rd_series=file_rd_series,
                             file_wr_series=file_wr_series,
                             ax=ax_files,
                             log_filename=log_file)

        fig.set_size_inches(12, 4)
        fig.tight_layout()
        fig.savefig(f'{log_file}_data_access_by_category.png', dpi=300)
