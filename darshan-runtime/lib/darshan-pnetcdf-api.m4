dnl Process this m4 file to produce 'C' language file.
dnl
dnl If you see this line, you can ignore the next one.
/* Do not edit this file. It is produced from the corresponding .m4 source */
dnl
/*
 *  See COPYRIGHT notice in top-level directory.
 */

divert(`-1')
# foreach(x, (item_1, item_2, ..., item_n), stmt)
#   parenthesized list, simple version
define(`foreach', `pushdef(`$1')_foreach($@)popdef(`$1')')
define(`_arg1', `$1')
define(`_foreach', `ifelse(`$2', `()', `',
  `define(`$1', _arg1$2)$3`'$0(`$1', (shift$2), `$3')')')
divert`'dnl

define(`Upcase',`translit($1, abcdefghijklmnopqrstuvwxyz, ABCDEFGHIJKLMNOPQRSTUVWXYZ)')dnl

dnl
dnl translate type name to C data type
dnl
define(`NC2ITYPE', `ifelse(`$1', `_text',      `char',
                           `$1', `_schar',     `signed char',
                           `$1', `_uchar',     `unsigned char',
                           `$1', `_short',     `short',
                           `$1', `_ushort',    `unsigned short',
                           `$1', `_int',       `int',
                           `$1', `_uint',      `unsigned int',
                           `$1', `_long',      `long',
                           `$1', `_float',     `float',
                           `$1', `_double',    `double',
                           `$1', `_longlong',  `long long',
                           `$1', `_ulonglong', `unsigned long long')')dnl
dnl
dnl List of type names used in PnetCDF APIs
dnl
define(`ITYPE_LIST', `_text, _schar, _uchar, _short, _ushort, _int, _uint, _long, _float, _double, _longlong, _ulonglong')dnl
dnl
dnl API arguments related to buffer, buffer type, and buffer count
dnl
define(`BufArgs', `ifelse(`$2', `',
                          `ifelse($1, `get', `void *buf,',
                                  $1,`iget', `void *buf,', `const void *buf,')
                          MPI_Offset bufcount, MPI_Datatype buftype',
                          `ifelse($1, `get', `NC2ITYPE($2) *buf',
                                  $1,`iget', `NC2ITYPE($2) *buf',
                                             `const NC2ITYPE($2) *buf')')')dnl
dnl
dnl start/count/stride/imap arguments for different kinds of APIs
dnl
define(`ArgKind', `ifelse(
       `$1', `1', `const MPI_Offset *start,',
       `$1', `a', `const MPI_Offset *start, const MPI_Offset *count,',
       `$1', `s', `const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride,',
       `$1', `m', `const MPI_Offset *start, const MPI_Offset *count, const MPI_Offset *stride, const MPI_Offset *imap,')')dnl
dnl
define(`ArgKindName', `ifelse(
       `$1', `1', `start,',
       `$1', `a', `start, count,',
       `$1', `s', `start, count, stride,',
       `$1', `m', `start, count, stride, imap,')')dnl

dnl
define(`UPDATE_GETPUT_COUNTERS',`ifelse(
`$1',`get',`rec_ref->var_rec->counters[PNETCDF_VAR_BYTES_READ] += $3;
            rec_ref->var_rec->counters[`PNETCDF_VAR_GET_VAR'Upcase($2)] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_WRITE)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_READ;
            DARSHAN_BUCKET_INC(
                &(rec_ref->var_rec->counters[PNETCDF_VAR_SIZE_READ_AGG_0_100]), $3);
            common_access_vals[0] = $3;
            cvc = darshan_track_common_val_counters(&rec_ref->access_root,
                common_access_vals, PNETCDF_VAR_MAX_NDIMS+PNETCDF_VAR_MAX_NDIMS+1, &rec_ref->access_count);
            if (cvc) DARSHAN_UPDATE_COMMON_VAL_COUNTERS(
                &(rec_ref->var_rec->counters[PNETCDF_VAR_ACCESS1_ACCESS]),
                &(rec_ref->var_rec->counters[PNETCDF_VAR_ACCESS1_COUNT]),
                cvc->vals, cvc->nvals, cvc->freq, 0);
            if (rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_START_TIMESTAMP] == 0 ||
             rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_START_TIMESTAMP] > tm1)
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_START_TIMESTAMP] = tm1;
            rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_END_TIMESTAMP] = tm2;
            if (rec_ref->var_rec->fcounters[PNETCDF_VAR_F_MAX_READ_TIME] < tm2 - tm1) {
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_MAX_READ_TIME] = tm2 - tm1;
                rec_ref->var_rec->counters[PNETCDF_VAR_MAX_READ_TIME_SIZE] = $3;
            }
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_TIME],
                tm1, tm2, rec_ref->last_read_end);',
`$1',`put',`rec_ref->var_rec->counters[PNETCDF_VAR_BYTES_WRITTEN] += $3;
            rec_ref->var_rec->counters[`PNETCDF_VAR_PUT_VAR'Upcase($2)] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_READ)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_WRITE;
            DARSHAN_BUCKET_INC(
                &(rec_ref->var_rec->counters[PNETCDF_VAR_SIZE_WRITE_AGG_0_100]), $3);
            common_access_vals[0] = $3;
            cvc = darshan_track_common_val_counters(&rec_ref->access_root,
                common_access_vals, PNETCDF_VAR_MAX_NDIMS+PNETCDF_VAR_MAX_NDIMS+1, &rec_ref->access_count);
            if (cvc) DARSHAN_UPDATE_COMMON_VAL_COUNTERS(
                &(rec_ref->var_rec->counters[PNETCDF_VAR_ACCESS1_ACCESS]),
                &(rec_ref->var_rec->counters[PNETCDF_VAR_ACCESS1_COUNT]),
                cvc->vals, cvc->nvals, cvc->freq, 0);
            if (rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_START_TIMESTAMP] == 0 ||
             rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_START_TIMESTAMP] > tm1)
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_START_TIMESTAMP] = tm1;
            rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_END_TIMESTAMP] = tm2;
            if (rec_ref->var_rec->fcounters[PNETCDF_VAR_F_MAX_WRITE_TIME] < tm2 - tm1) {
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_MAX_WRITE_TIME] = tm2 - tm1;
                rec_ref->var_rec->counters[PNETCDF_VAR_MAX_WRITE_TIME_SIZE] = access_size;
            }
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_TIME],
                tm1, tm2, rec_ref->last_write_end);')')dnl
dnl
dnl
define(`UPDATE_INDEPCOLL_RW_COUNTER',`ifelse(
       `$1$2',`get',`rec_ref->var_rec->counters[PNETCDF_VAR_INDEP_READS] += 1',
       `$1$2',`get_all',`rec_ref->var_rec->counters[PNETCDF_VAR_COLL_READS] += 1',
       `$1$2',`put',`rec_ref->var_rec->counters[PNETCDF_VAR_INDEP_WRITES] += 1',
       `$1$2',`put_all',`rec_ref->var_rec->counters[PNETCDF_VAR_COLL_WRITES] += 1')')dnl
dnl

dnl
define(`APINAME',`ifelse(`$3',`',`ncmpi_$1_var$2$4',`ncmpi_$1_var$2$3$4')')dnl
dnl

/* ncmpi_get/put_var<kind>_<type>_<mode> API */
dnl
dnl GETPUT_API(get/put, `'/1/a/s/m, `'/itype, `'/_all)
dnl
define(`GETPUT_API',`
DARSHAN_FORWARD_DECL(APINAME($1,$2,$3,$4), int, (int ncid, int varid, ArgKind($2) BufArgs($1,$3)));

int DARSHAN_DECL(APINAME($1,$2,$3,$4))(int ncid, int varid, ArgKind($2)BufArgs($1,$3))
{
    int err, ret;
    MPI_Offset $1_before, $1_after;
    double tm1, tm2;

    MAP_OR_FAIL(APINAME($1,$2,$3,$4));

    err = ncmpi_inq_$1_size(ncid, &$1_before);
    if (err != NC_NOERR) return err;

    tm1 = darshan_core_wtime();
    ret = `__real_'APINAME($1,$2,$3,$4)(ncid, varid, ArgKindName($2) buf ifelse($3,`',`, bufcount, buftype'));
    tm2 = darshan_core_wtime();

    err = ncmpi_inq_$1_size(ncid, &$1_after);
    if (err != NC_NOERR) return err;

    if (ret == NC_NOERR) {
        struct darshan_common_val_counter *cvc;
        int64_t common_access_vals[PNETCDF_VAR_MAX_NDIMS+PNETCDF_VAR_MAX_NDIMS+1] = {0};

        PNETCDF_VAR_PRE_RECORD();
        struct pnetcdf_var_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            size_t access_size = $1_after - $1_before;
            UPDATE_GETPUT_COUNTERS($1,$2,access_size)
            UPDATE_INDEPCOLL_RW_COUNTER($1,$4);
        }
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
')dnl

foreach(`kind', (, 1, a, s, m),
        `foreach(`putget', (put, get),
                 `foreach(`collindep', (, _all),
                        `foreach(`iType', (`',ITYPE_LIST),
                                   `GETPUT_API(putget,kind,iType,collindep)'
)')')')dnl

/* ncmpi_get/put_varn_<type>_<mode> API */
dnl
dnl GETPUT_VARN_API(get/put, `'/itype, `'/_all)
dnl
define(`GETPUT_VARN_API',`
DARSHAN_FORWARD_DECL(APINAME($1,n,$2,$3), int, (int ncid, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, BufArgs($1,$2)));

int DARSHAN_DECL(APINAME($1,n,$2,$3))(int ncid, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, BufArgs($1,$2))
{
    int err, ret;
    MPI_Offset $1_before, $1_after;
    double tm1, tm2;

    MAP_OR_FAIL(APINAME($1,n,$2,$3));

    err = ncmpi_inq_$1_size(ncid, &$1_before);
    if (err != NC_NOERR) return err;

    tm1 = darshan_core_wtime();
    ret = `__real_'APINAME($1,n,$2,$3)(ncid, varid, num, starts, counts, buf ifelse($2,`',`, bufcount, buftype'));
    tm2 = darshan_core_wtime();

    err = ncmpi_inq_$1_size(ncid, &$1_after);
    if (err != NC_NOERR) return err;

    if (ret == NC_NOERR) {
        struct darshan_common_val_counter *cvc;
        int64_t common_access_vals[PNETCDF_VAR_MAX_NDIMS+PNETCDF_VAR_MAX_NDIMS+1] = {0};

        PNETCDF_VAR_PRE_RECORD();
        struct pnetcdf_var_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            size_t access_size = $1_after - $1_before;
            UPDATE_GETPUT_COUNTERS($1,N,access_size)
            UPDATE_INDEPCOLL_RW_COUNTER($1,$4);
        }
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
')dnl

foreach(`putget', (put, get),
        `foreach(`collindep', (, _all),
                 `foreach(`iType', (`',ITYPE_LIST),
                          `GETPUT_VARN_API(putget,iType,collindep)'
)')')dnl

/* ncmpi_get/put_vard_<type>_<mode> API */
dnl
dnl GETPUT_VARD_API(get/put, `'/itype, `'/_all)
dnl
define(`GETPUT_VARD_API',`
DARSHAN_FORWARD_DECL(ncmpi_$1_vard$2, int, (int ncid, int varid, MPI_Datatype filetype, ifelse($1,`put',`const ')void *buf, MPI_Offset bufcount, MPI_Datatype buftype));

int DARSHAN_DECL(ncmpi_$1_vard$2)(int ncid, int varid, MPI_Datatype filetype, ifelse($1,`put',`const ')void *buf, MPI_Offset bufcount, MPI_Datatype buftype)
{
    int err, ret;
    MPI_Offset $1_before, $1_after;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_$1_vard$2);

    err = ncmpi_inq_$1_size(ncid, &$1_before);
    if (err != NC_NOERR) return err;

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_$1_vard$2(ncid, varid, filetype, buf, bufcount, buftype);
    tm2 = darshan_core_wtime();

    err = ncmpi_inq_$1_size(ncid, &$1_after);
    if (err != NC_NOERR) return err;

    if (ret == NC_NOERR) {
        struct darshan_common_val_counter *cvc;
        int64_t common_access_vals[PNETCDF_VAR_MAX_NDIMS+PNETCDF_VAR_MAX_NDIMS+1] = {0};

        PNETCDF_VAR_PRE_RECORD();
        struct pnetcdf_var_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            size_t access_size = $1_after - $1_before;
            UPDATE_GETPUT_COUNTERS($1,D,access_size)
            UPDATE_INDEPCOLL_RW_COUNTER($1,$4);
        }
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
')dnl

foreach(`putget', (put, get),
        `foreach(`collindep', (, _all),
                 `GETPUT_VARD_API(putget,collindep)'
)')dnl

/* ncmpi_iget/iput/bput_var<kind>_<type> API */
dnl
dnl IGETPUT_API(iget/iput/bput, `'/1/a/s/m, `'/itype)
dnl
define(`IGETPUT_API',`
DARSHAN_FORWARD_DECL(APINAME($1,$2,$3), int, (int ncid, int varid, ArgKind($2)BufArgs($1,$3), int *reqid));

int DARSHAN_DECL(APINAME($1,$2,$3))(int ncid, int varid, ArgKind($2)BufArgs($1,$3), int *reqid)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(APINAME($1,$2,$3));

    tm1 = darshan_core_wtime();
    ret = `__real_'APINAME($1,$2,$3)(ncid, varid, ArgKindName($2) buf ifelse($3,`',`, bufcount, buftype'), reqid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_VAR_PRE_RECORD();
        struct pnetcdf_var_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->var_rec->counters[`PNETCDF_VAR_'Upcase($1)`_VAR'Upcase($2)] += 1;
            ifelse($1,`iget',
           `rec_ref->var_rec->counters[PNETCDF_VAR_NB_READS] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_WRITE)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_READ;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_TIME],
                tm1, tm2, rec_ref->last_read_end);',
            `rec_ref->var_rec->counters[PNETCDF_VAR_NB_WRITES] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_READ)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_WRITE;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_TIME],
                tm1, tm2, rec_ref->last_write_end);')
        }
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
')dnl

foreach(`kind', (, 1, a, s, m),
        `foreach(`putget', (iput, iget, bput),
                 `foreach(`iType', (`',ITYPE_LIST),
                          `IGETPUT_API(putget,kind,iType)'
)')')dnl

/* ncmpi_iget/iput/bput_varn_<type> API */
dnl
dnl IGETPUT_VARN_API(iget/iput/bput, `'/itype)
dnl
define(`IGETPUT_VARN_API',`
DARSHAN_FORWARD_DECL(APINAME($1,n,$2), int, (int ncid, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, BufArgs($1,$2), int *reqid));

int DARSHAN_DECL(APINAME($1,n,$2))(int ncid, int varid, int num, MPI_Offset* const *starts, MPI_Offset* const *counts, BufArgs($1,$2), int *reqid)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(APINAME($1,n,$2));

    tm1 = darshan_core_wtime();
    ret = `__real_'APINAME($1,n,$2)(ncid, varid, num, starts, counts, buf, ifelse($2,`',`bufcount, buftype,') reqid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_VAR_PRE_RECORD();
        struct pnetcdf_var_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->var_rec->counters[`PNETCDF_VAR_'Upcase($1)`_VARN'] += 1;
            ifelse($1,`iget',
           `rec_ref->var_rec->counters[PNETCDF_VAR_NB_READS] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_WRITE)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_READ;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_READ_TIME],
                tm1, tm2, rec_ref->last_read_end);',
            `rec_ref->var_rec->counters[PNETCDF_VAR_NB_WRITES] += 1;
            if (rec_ref->last_io_type == DARSHAN_IO_READ)
                rec_ref->var_rec->counters[PNETCDF_VAR_RW_SWITCHES] += 1;
            rec_ref->last_io_type = DARSHAN_IO_WRITE;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->var_rec->fcounters[PNETCDF_VAR_F_WRITE_TIME],
                tm1, tm2, rec_ref->last_write_end);')
        }
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
')dnl

foreach(`putget', (iput, iget, bput),
        `foreach(`iType', (`',ITYPE_LIST),
                 `IGETPUT_VARN_API(putget,iType)'
)')dnl

dnl
dnl PNETCDF_FILE_RECORD(create/open, ncidp, path, comm, tm1, tm2)
dnl
define(`PNETCDF_FILE_RECORD',`
    do {
        darshan_record_id rec_id;
        struct pnetcdf_file_record_ref *rec_ref;
        char *newpath;
        int comm_size;
        newpath = darshan_clean_file_path($3);
        if (!newpath) newpath = (char *)$3;
        if (darshan_core_excluded_path(newpath)) {
            if (newpath != $3) free(newpath);
            break;
        }
        rec_id = darshan_core_gen_record_id(newpath);
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->rec_id_hash, &rec_id, sizeof(darshan_record_id));
        if (!rec_ref) rec_ref = pnetcdf_file_track_new_record(rec_id, newpath);
        if (!rec_ref) {
            if (newpath != $3) free(newpath);
            break;
        }
        PMPI_Comm_size($4, &comm_size);
        if (rec_ref->file_rec->fcounters[`PNETCDF_F_'Upcase($1)`_START_TIMESTAMP'] == 0 ||
            rec_ref->file_rec->fcounters[`PNETCDF_F_'Upcase($1)`_START_TIMESTAMP'] > $5)
            rec_ref->file_rec->fcounters[`PNETCDF_F_'Upcase($1)`_START_TIMESTAMP'] = $5;
        rec_ref->file_rec->fcounters[`PNETCDF_F_'Upcase($1)`_END_TIMESTAMP'] = $6;
        rec_ref->file_rec->counters[`PNETCDF_'Upcase($1)`S'] += 1;
        DARSHAN_TIMER_INC_NO_OVERLAP(rec_ref->file_rec->fcounters[`PNETCDF_F_'Upcase($1)`_TIME'],
            tm1, tm2, rec_ref->last_$1_end);
        darshan_add_record_ref(&(pnetcdf_file_runtime->ncid_hash), $2, sizeof(int), rec_ref);
        if (newpath != $3) free(newpath);
    } while (0);
')dnl

DARSHAN_FORWARD_DECL(ncmpi_create, int, (MPI_Comm comm, const char *path, int cmode, MPI_Info info, int *ncidp));
int DARSHAN_DECL(ncmpi_create)(MPI_Comm comm, const char *path,
    int cmode, MPI_Info info, int *ncidp)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_create);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_create(comm, path, cmode, info, ncidp);
    tm2 = darshan_core_wtime();
    if (ret == NC_NOERR)
    { /* use ROMIO approach to strip prefix if present */
        /* strip off prefix if there is one, but only skip prefixes
         * if they are greater than length one to allow for windows
         * drive specifications (e.g. c:\...)
         */
        char* tmp = strchr(path, ':');
        if (tmp > path + 1) {
            path = tmp + 1;
        }

        PNETCDF_FILE_PRE_RECORD();
        PNETCDF_FILE_RECORD(create, ncidp, path, comm, tm1, tm2)
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_open, int, (MPI_Comm comm, const char *path, int omode, MPI_Info info, int *ncidp));

int DARSHAN_DECL(ncmpi_open)(MPI_Comm comm, const char *path,
    int omode, MPI_Info info, int *ncidp)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_open);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_open(comm, path, omode, info, ncidp);
    tm2 = darshan_core_wtime();
    if (ret == NC_NOERR) {
        /* use ROMIO approach to strip prefix if present */
        /* strip off prefix if there is one, but only skip prefixes
         * if they are greater than length one to allow for windows
         * drive specifications (e.g. c:\...)
         */
        char* tmp = strchr(path, ':');
        if (tmp > path + 1) {
            path = tmp + 1;
        }

        PNETCDF_FILE_PRE_RECORD();
        PNETCDF_FILE_RECORD(open, ncidp, path, comm, tm1, tm2)
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_close, int, (int ncid));

int DARSHAN_DECL(ncmpi_close)(int ncid)
{
    int ret;
    double tm1, tm2;
    MPI_Offset put_size, get_size;
    struct pnetcdf_file_record_ref *rec_ref;

    MAP_OR_FAIL(ncmpi_close);

    ret = ncmpi_inq_put_size(ncid, &put_size);
    if (ret != NC_NOERR) return ret;
    ret = ncmpi_inq_get_size(ncid, &get_size);
    if (ret != NC_NOERR) return ret;

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_close(ncid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash,
            &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->file_rec->counters[PNETCDF_BYTES_WRITTEN] += put_size;
            rec_ref->file_rec->counters[PNETCDF_BYTES_READ] += get_size;

            if (rec_ref->file_rec->fcounters[PNETCDF_F_CLOSE_START_TIMESTAMP] == 0 ||
             rec_ref->file_rec->fcounters[PNETCDF_F_CLOSE_START_TIMESTAMP] > tm1)
               rec_ref->file_rec->fcounters[PNETCDF_F_CLOSE_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_CLOSE_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_CLOSE_TIME],
                tm1, tm2, rec_ref->last_close_end);
            darshan_delete_record_ref(&(pnetcdf_file_runtime->ncid_hash),
                &ncid, sizeof(int));
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_enddef, int, (int ncid));

int DARSHAN_DECL(ncmpi_enddef)(int ncid)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_enddef);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_enddef(ncid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        struct pnetcdf_file_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->file_rec->counters[PNETCDF_ENDDEFS] += 1;
            if (rec_ref->file_rec->fcounters[PNETCDF_F_ENDDEF_START_TIMESTAMP] == 0 ||
                rec_ref->file_rec->fcounters[PNETCDF_F_ENDDEF_START_TIMESTAMP] > tm1)
                rec_ref->file_rec->fcounters[PNETCDF_F_ENDDEF_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_ENDDEF_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_ENDDEF_TIME],
                tm1, tm2, rec_ref->last_enddef_end);
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_redef, int, (int ncid));

int DARSHAN_DECL(ncmpi_redef)(int ncid)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_redef);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_redef(ncid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        struct pnetcdf_file_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->file_rec->counters[PNETCDF_REDEFS] += 1;
            if (rec_ref->file_rec->fcounters[PNETCDF_F_REDEF_START_TIMESTAMP] == 0 ||
             rec_ref->file_rec->fcounters[PNETCDF_F_REDEF_START_TIMESTAMP] > tm1)
                rec_ref->file_rec->fcounters[PNETCDF_F_REDEF_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_REDEF_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_REDEF_TIME],
                tm1, tm2, rec_ref->last_redef_end);
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_wait, int, (int ncid, int num, int array_of_requests[], int array_of_statuses[]));

int DARSHAN_DECL(ncmpi_wait)(int ncid, int num, int array_of_requests[],
    int array_of_statuses[])
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_wait);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_wait(ncid, num, array_of_requests, array_of_statuses);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        struct pnetcdf_file_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->file_rec->counters[PNETCDF_INDEP_WAITS] += 1;
            if (rec_ref->file_rec->fcounters[PNETCDF_F_INDEP_WAIT_START_TIMESTAMP] == 0 ||
                rec_ref->file_rec->fcounters[PNETCDF_F_INDEP_WAIT_START_TIMESTAMP] > tm1)
                rec_ref->file_rec->fcounters[PNETCDF_F_INDEP_WAIT_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_INDEP_WAIT_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_INDEP_WAIT_TIME],
                tm1, tm2, rec_ref->last_wait_indep_end);
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_wait_all, int, (int ncid, int num, int array_of_requests[], int array_of_statuses[]));

int DARSHAN_DECL(ncmpi_wait_all)(int ncid, int num, int array_of_requests[],
    int array_of_statuses[])
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_wait_all);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_wait_all(ncid, num, array_of_requests, array_of_statuses);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        struct pnetcdf_file_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref) {
            rec_ref->file_rec->counters[PNETCDF_COLL_WAITS] += 1;
            if (rec_ref->file_rec->fcounters[PNETCDF_F_COLL_WAIT_START_TIMESTAMP] == 0 ||
                rec_ref->file_rec->fcounters[PNETCDF_F_COLL_WAIT_START_TIMESTAMP] > tm1)
                rec_ref->file_rec->fcounters[PNETCDF_F_COLL_WAIT_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_COLL_WAIT_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_COLL_WAIT_TIME],
                tm1, tm2, rec_ref->last_wait_coll_end);
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_sync, int, (int ncid));

int DARSHAN_DECL(ncmpi_sync)(int ncid)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_sync);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_sync(ncid);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_FILE_PRE_RECORD();
        struct pnetcdf_file_record_ref *rec_ref;
        rec_ref = darshan_lookup_record_ref(pnetcdf_file_runtime->ncid_hash, &ncid, sizeof(int));
        if (rec_ref)
        {
            rec_ref->file_rec->counters[PNETCDF_SYNCS] += 1;
            if (rec_ref->file_rec->fcounters[PNETCDF_F_SYNC_START_TIMESTAMP] == 0 ||
                rec_ref->file_rec->fcounters[PNETCDF_F_SYNC_START_TIMESTAMP] > tm1)
                rec_ref->file_rec->fcounters[PNETCDF_F_SYNC_START_TIMESTAMP] = tm1;
            rec_ref->file_rec->fcounters[PNETCDF_F_SYNC_END_TIMESTAMP] = tm2;
            DARSHAN_TIMER_INC_NO_OVERLAP(
                rec_ref->file_rec->fcounters[PNETCDF_F_SYNC_TIME],
                tm1, tm2, rec_ref->last_sync_end);
        }
        PNETCDF_FILE_POST_RECORD();
    }
    return(ret);
}

dnl
dnl PNETCDF_VAR_RECORD_OPEN(ncid, name, xtype, ndims, dimids, varidp, tm1, tm2)
dnl
define(`PNETCDF_VAR_RECORD_OPEN',`
    do {
        char *file_path, *rec_name;
        int err, i, ret_name_len, unlimdimid, type_size;
        darshan_record_id rec_id;
        struct pnetcdf_var_record_ref *rec_ref;
        MPI_Offset npoints;

        /* get corresponding file name */
        err = ncmpi_inq_path($1, &ret_name_len, NULL);
        if (err != NC_NOERR) break;
        rec_name = (char*) malloc(ret_name_len+1);
        err = ncmpi_inq_path($1, NULL, rec_name);
        if (err != NC_NOERR) { free(rec_name); break; }
        /* fully resolve file path */
        file_path = darshan_clean_file_path(rec_name);
        if (darshan_core_excluded_path(file_path)) {
            free(file_path);
            free(rec_name);
            break;
        }
        free(rec_name);
        rec_name = (char*) malloc(strlen(file_path)+strlen($2)+2);
        strcpy(rec_name, file_path);
        free(file_path);
        /* append variable name if we have space */
        strcat(rec_name, DARSHAN_PNETCDF_DATASET_DELIM);
        strcat(rec_name, $2);
        rec_id = darshan_core_gen_record_id(rec_name);
        rec_ref = darshan_lookup_record_ref(pnetcdf_var_runtime->rec_id_hash, &rec_id, sizeof(darshan_record_id));
        if (!rec_ref) rec_ref = pnetcdf_var_track_new_record(rec_id, rec_name);
        free(rec_name);
        if (!rec_ref) break;
        rec_ref->var_rec->counters[PNETCDF_VAR_OPENS] += 1;
        if (rec_ref->var_rec->fcounters[PNETCDF_VAR_F_OPEN_START_TIMESTAMP] == 0 ||
            rec_ref->var_rec->fcounters[PNETCDF_VAR_F_OPEN_START_TIMESTAMP] > $7)
            rec_ref->var_rec->fcounters[PNETCDF_VAR_F_OPEN_START_TIMESTAMP] = $7;
        rec_ref->var_rec->fcounters[PNETCDF_VAR_F_OPEN_END_TIMESTAMP] = $8;
        DARSHAN_TIMER_INC_NO_OVERLAP(rec_ref->var_rec->fcounters[PNETCDF_VAR_F_META_TIME],
            $7, $8, rec_ref->last_meta_end);
        err = ncmpi_inq_unlimdim($1, &unlimdimid);
        i = ($5[0] == unlimdimid) ? 1 : 0; /* record variable or not */
        for (npoints = 1; i < $4; i++) {
            MPI_Offset dim_len;
            err = ncmpi_inq_dimlen($1, $5[i], &dim_len);
            npoints *= dim_len;
        }
        rec_ref->var_rec->counters[PNETCDF_VAR_NPOINTS] = npoints;
        rec_ref->var_rec->counters[PNETCDF_VAR_NDIMS] = $4;
        if ($3 == NC_BYTE || $3 == NC_CHAR || $3 == NC_UBYTE) type_size = 1;
        else if ($3 == NC_SHORT || $3 == NC_USHORT) type_size = 2;
        else if ($3 == NC_INT || $3 == NC_UINT || $3 == NC_FLOAT) type_size = 4;
        else type_size = 8;
        rec_ref->var_rec->counters[PNETCDF_VAR_DATATYPE_SIZE] = type_size;
        darshan_add_record_ref(&(pnetcdf_var_runtime->ncid_hash), $6, sizeof(int), rec_ref);
    } while (0);
')dnl

DARSHAN_FORWARD_DECL(ncmpi_def_var, int, (int ncid, const char *name, nc_type xtype, int ndims, const int dimids[], int *varidp));

int DARSHAN_DECL(ncmpi_def_var)(int ncid, const char *name, nc_type xtype, int ndims, const int dimids[], int *varidp)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_def_var);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_def_var(ncid, name, xtype, ndims, dimids, varidp);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        PNETCDF_VAR_PRE_RECORD();
        PNETCDF_VAR_RECORD_OPEN(ncid, name, xtype, ndims, dimids, varidp, tm1, tm2);
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}

DARSHAN_FORWARD_DECL(ncmpi_inq_varid, int, (int ncid, const char *name, int *varidp));

int DARSHAN_DECL(ncmpi_inq_varid)(int ncid, const char *name, int *varidp)
{
    int ret;
    double tm1, tm2;

    MAP_OR_FAIL(ncmpi_inq_varid);

    tm1 = darshan_core_wtime();
    ret = __real_ncmpi_inq_varid(ncid, name, varidp);
    tm2 = darshan_core_wtime();

    if (ret == NC_NOERR) {
        nc_type xtype;
        int ndims, *dimids;
        ret = ncmpi_inq_vartype(ncid, *varidp, &xtype);
        ret = ncmpi_inq_varndims(ncid, *varidp, &ndims);
        dimids = (int*) malloc(ndims * sizeof(int));
        ret = ncmpi_inq_vardimid(ncid, *varidp, dimids);

        PNETCDF_VAR_PRE_RECORD();
        PNETCDF_VAR_RECORD_OPEN(ncid, name, xtype, ndims, dimids, varidp, tm1, tm2);
        free(dimids);
        PNETCDF_VAR_POST_RECORD();
    }
    return(ret);
}
