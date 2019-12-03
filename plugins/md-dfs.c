// This file is part of MD-REAL-IO.
//
// MD-REAL-IO is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MD-REAL-IO is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MD-REAL-IO.  If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2019 Intel Corporation

#include <string.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <libgen.h>

#include <gurt/common.h>
#include <gurt/hash.h>
#include <daos.h>
#include <daos_fs.h>
#include <mpi.h>
#include <plugins/md-dfs.h>

static char *pool = NULL;
static char *cont = NULL;
static char *svcl = NULL;
static char *prefix = NULL;
static char *dir = "out";
static char *oclass = NULL;
static daos_size_t chunk_size = 0;
static int destroy = 0;

dfs_t *dfs;
static daos_handle_t poh, coh;
static daos_oclass_id_t objectClass = OC_S1;
static struct d_hash_table *dir_hash;
int rank;

struct aiori_dir_hdl {
        d_list_t	entry;
        dfs_obj_t	*oh;
        char		name[PATH_MAX];
};

enum handleType {
        POOL_HANDLE,
        CONT_HANDLE,
	ARRAY_HANDLE
};

static option_help options [] = {
  {'D', "dfs.prefix", "DFS_PREFIX", OPTION_OPTIONAL_ARGUMENT, 's', &prefix},
  {'R', "dfs.root", "DFS_ROOT_DIR", OPTION_OPTIONAL_ARGUMENT, 's', &dir},
  {'P', "dfs.pool", "POOL UUID", OPTION_OPTIONAL_ARGUMENT, 's', &pool},
  {'C', "dfs.cont", "Container UUID", OPTION_OPTIONAL_ARGUMENT, 's', &cont},
  {'S', "dfs.svcl", "POOL SVCL", OPTION_OPTIONAL_ARGUMENT, 's', &svcl},
  {'N', "dfs.chunk_size", "chunk size", OPTION_OPTIONAL_ARGUMENT, 'd', &chunk_size},
  {'O', "dfs.oclass", "object class", OPTION_OPTIONAL_ARGUMENT, 's', &oclass},
  {'d', "dfs.destroy", "Destroy DFS Container", OPTION_FLAG, 'd', &destroy},
  LAST_OPTION
};

static option_help * get_options(){
  return options;
}

/* For DAOS methods. */
#define DCHECK(rc, ...)							\
do {                                                                    \
        int _rc = (rc);                                                 \
                                                                        \
        if (_rc != 0) {							\
	  fprintf(stderr, "ERROR (%s:%d): %d: %d:\n",			\
		  __FILE__, __LINE__, rank, _rc);			\
	  fprintf(stderr, __VA_ARGS__);					\
                fflush(stderr);                                         \
                exit(-1);                                       	\
        }                                                               \
} while (0)

static inline struct aiori_dir_hdl *
hdl_obj(d_list_t *rlink)
{
        return container_of(rlink, struct aiori_dir_hdl, entry);
}

static bool
key_cmp(struct d_hash_table *htable, d_list_t *rlink,
	const void *key, unsigned int ksize)
{
        struct aiori_dir_hdl *hdl = hdl_obj(rlink);

        return (strcmp(hdl->name, (const char *)key) == 0);
}

static void
rec_free(struct d_hash_table *htable, d_list_t *rlink)
{
        struct aiori_dir_hdl *hdl = hdl_obj(rlink);

        assert(d_hash_rec_unlinked(&hdl->entry));
        dfs_release(hdl->oh);
        free(hdl);
}

static d_hash_table_ops_t hdl_hash_ops = {
        .hop_key_cmp	= key_cmp,
        .hop_rec_free	= rec_free
};

/* Distribute process 0's pool or container handle to others. */
static void
HandleDistribute(daos_handle_t *handle, enum handleType type)
{
        d_iov_t global;
        int        rc;

        global.iov_buf = NULL;
        global.iov_buf_len = 0;
        global.iov_len = 0;

        assert(type == POOL_HANDLE || type == CONT_HANDLE);
        if (rank == 0) {
                /* Get the global handle size. */
                if (type == POOL_HANDLE)
                        rc = daos_pool_local2global(*handle, &global);
                else
                        rc = daos_cont_local2global(*handle, &global);
                DCHECK(rc, "Failed to get global handle size");
        }

        MPI_Bcast(&global.iov_buf_len, 1, MPI_UINT64_T, 0,
		  MPI_COMM_WORLD);

	global.iov_len = global.iov_buf_len;
        global.iov_buf = malloc(global.iov_buf_len);
        if (global.iov_buf == NULL) {
		fprintf(stderr, "failed to alloc global handle buffer\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

        if (rank == 0) {
                if (type == POOL_HANDLE)
                        rc = daos_pool_local2global(*handle, &global);
                else
                        rc = daos_cont_local2global(*handle, &global);
                DCHECK(rc, "Failed to create global handle");
        }

        MPI_Bcast(global.iov_buf, global.iov_buf_len, MPI_BYTE, 0,
		  MPI_COMM_WORLD);

        if (rank != 0) {
                if (type == POOL_HANDLE)
                        rc = daos_pool_global2local(global, handle);
                else
                        rc = daos_cont_global2local(poh, global, handle);
                DCHECK(rc, "Failed to get local handle");
        }

        free(global.iov_buf);
}

static int
parse_filename(const char *path, char **_obj_name, char **_cont_name)
{
	char *f1 = NULL;
	char *f2 = NULL;
	char *fname = NULL;
	char *cont_name = NULL;
	int rc = 0;

	if (path == NULL || _obj_name == NULL || _cont_name == NULL)
		return -EINVAL;

	if (strcmp(path, "/") == 0) {
		*_cont_name = strdup("/");
		if (*_cont_name == NULL)
			return -ENOMEM;
		*_obj_name = NULL;
		return 0;
	}

	f1 = strdup(path);
	if (f1 == NULL) {
                rc = -ENOMEM;
                goto out;
        }

	f2 = strdup(path);
	if (f2 == NULL) {
                rc = -ENOMEM;
                goto out;
        }

	fname = basename(f1);
	cont_name = dirname(f2);

	if (cont_name[0] == '.' || cont_name[0] != '/') {
		char cwd[1024];

		if (getcwd(cwd, 1024) == NULL) {
                        rc = -ENOMEM;
                        goto out;
                }

		if (strcmp(cont_name, ".") == 0) {
			cont_name = strdup(cwd);
			if (cont_name == NULL) {
                                rc = -ENOMEM;
                                goto out;
                        }
		} else {
			char *new_dir = calloc(strlen(cwd) + strlen(cont_name)
					       + 1, sizeof(char));
			if (new_dir == NULL) {
                                rc = -ENOMEM;
                                goto out;
                        }

			strcpy(new_dir, cwd);
			if (cont_name[0] == '.') {
				strcat(new_dir, &cont_name[1]);
			} else {
				strcat(new_dir, "/");
				strcat(new_dir, cont_name);
			}
			cont_name = new_dir;
		}
		*_cont_name = cont_name;
	} else {
		*_cont_name = strdup(cont_name);
		if (*_cont_name == NULL) {
                        rc = -ENOMEM;
                        goto out;
                }
	}

	*_obj_name = strdup(fname);
	if (*_obj_name == NULL) {
		free(*_cont_name);
		*_cont_name = NULL;
                rc = -ENOMEM;
                goto out;
	}

out:
	if (f1)
		free(f1);
	if (f2)
		free(f2);
	return rc;
}

static dfs_obj_t *
lookup_insert_dir(const char *name)
{
        struct aiori_dir_hdl *hdl;
        d_list_t *rlink;
        int rc;

        rlink = d_hash_rec_find(dir_hash, name, strlen(name));
        if (rlink != NULL) {
                hdl = hdl_obj(rlink);
                return hdl->oh;
        }

        hdl = calloc(1, sizeof(struct aiori_dir_hdl));
        if (hdl == NULL) {
		fprintf(stderr, "failed to alloc dir handle\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

        strncpy(hdl->name, name, PATH_MAX-1);
        hdl->name[PATH_MAX-1] = '\0';

        rc = dfs_lookup(dfs, name, O_RDWR, &hdl->oh, NULL, NULL);
        if (rc)
                return NULL;

        rc = d_hash_rec_insert(dir_hash, hdl->name, strlen(hdl->name),
                               &hdl->entry, true);
        DCHECK(rc, "Failed to insert dir handle in hashtable");

        return hdl->oh;
}

static int initialize()
{
	int rc;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (pool == NULL || svcl == NULL || cont == NULL) {
		fprintf(stderr, "Invalid pool or cont handles\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

        if (oclass) {
                objectClass = daos_oclass_name2id(oclass);
		if (objectClass == OC_UNKNOWN) {
			fprintf(stderr, "Invalid DAOS Object class %s\n", oclass);
			MPI_Abort(MPI_COMM_WORLD, -1);
		}
	}

	rc = daos_init();
        DCHECK(rc, "Failed to initialize daos");

        rc = d_hash_table_create(0, 16, NULL, &hdl_hash_ops, &dir_hash);
        DCHECK(rc, "Failed to initialize dir hashtable");

        if (rank == 0) {
                uuid_t pool_uuid, co_uuid;
                d_rank_list_t *svc = NULL;
                daos_pool_info_t pool_info;
                daos_cont_info_t co_info;

                rc = uuid_parse(pool, pool_uuid);
                DCHECK(rc, "Failed to parse 'Pool uuid': %s", pool);

                rc = uuid_parse(cont, co_uuid);
                DCHECK(rc, "Failed to parse 'Cont uuid': %s", cont);

                svc = daos_rank_list_parse(svcl, ":");
                if (svc == NULL) {
		  fprintf(stderr, "failed to parse svcl\n");
		  MPI_Abort(MPI_COMM_WORLD, -1);
		}

                printf("Pool uuid = %s, SVCL = %s\n", pool, svcl);
                printf("DFS Container namespace uuid = %s\n", cont);

                /** Connect to DAOS pool */
                rc = daos_pool_connect(pool_uuid, NULL, svc, DAOS_PC_RW,
                                       &poh, &pool_info, NULL);
                d_rank_list_free(svc);
                DCHECK(rc, "Failed to connect to pool");

                rc = daos_cont_open(poh, co_uuid, DAOS_COO_RW, &coh, &co_info,
                                    NULL);
                /* If NOEXIST we create it */
                if (rc == -DER_NONEXIST) {
                        printf("Creating DFS Container ...\n");

                        rc = dfs_cont_create(poh, co_uuid, NULL, &coh, NULL);
                        if (rc)
                                DCHECK(rc, "Failed to create container");
                } else if (rc) {
                        DCHECK(rc, "Failed to create container");
                }
        }

        HandleDistribute(&poh, POOL_HANDLE);
        HandleDistribute(&coh, CONT_HANDLE);

	rc = dfs_mount(poh, coh, O_RDWR, &dfs);
        DCHECK(rc, "Failed to mount DFS namespace");

        if (prefix) {
                rc = dfs_set_prefix(dfs, prefix);
                DCHECK(rc, "Failed to set DFS Prefix");
        }
	return MD_SUCCESS;
}

static int finalize()
{
        int rc;

	MPI_Barrier(MPI_COMM_WORLD);
        d_hash_table_destroy(dir_hash, true /* force */);

	rc = dfs_umount(dfs);
        DCHECK(rc, "Failed to umount DFS namespace");
	MPI_Barrier(MPI_COMM_WORLD);

	rc = daos_cont_close(coh, NULL);
        DCHECK(rc, "Failed to close container %s (%d)", cont, rc);
	MPI_Barrier(MPI_COMM_WORLD);

	if (destroy) {
                if (rank == 0) {
                        uuid_t uuid;
                        double t1, t2;

                        printf("Destorying DFS Container: %s\n", cont);
                        uuid_parse(cont, uuid);
                        t1 = MPI_Wtime();
                        rc = daos_cont_destroy(poh, uuid, 1, NULL);
                        t2 = MPI_Wtime();
                        if (rc == 0)
                                printf("Container Destroy time = %f secs", t2-t1);
                }

                MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (rc) {
			if (rank == 0)
				DCHECK(rc, "Failed to destroy container %s (%d)", cont, rc);
			MPI_Abort(MPI_COMM_WORLD, -1);
                }
        }

        if (rank == 0)
                printf("Disconnecting from DAOS POOL\n");

        rc = daos_pool_disconnect(poh, NULL);
        DCHECK(rc, "Failed to disconnect from pool");

	MPI_Barrier(MPI_COMM_WORLD);
	usleep(20000 * rank);

        if (rank == 0)
                printf("Finalizing DAOS..\n");

	rc = daos_fini();
        DCHECK(rc, "Failed to finalize DAOS");
	return MD_SUCCESS;
}

static int prepare_global(){
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL;
  int rc = 0;

  rc = parse_filename(dir, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", dir);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
	  fprintf(stderr, "Failed to lookup parent dir %s\n", dir_name);
	  return MD_ERROR_UNKNOWN;
  }

  rc = dfs_mkdir(dfs, parent, name, 0755);
  if (rc)
    fprintf(stderr, "dfs_mkdir() of %s Failed (%d)", dir, rc);

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return rc;
}

static int purge_global(){
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL;
  int rc = 0;

  rc = parse_filename(dir, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", dir);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
	  fprintf(stderr, "Failed to lookup parent dir %s\n", dir_name);
	  return MD_ERROR_UNKNOWN;
  }

  rc = dfs_remove(dfs, parent, name, false, NULL);
  if (rc)
    fprintf(stderr, "dfs_remove() of %s Failed (%d)", dir, rc);

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return rc;
}

static int def_dset_name(char * out_name, int n, int d){
  sprintf(out_name, "%s/%d_%d", dir, n, d);
  return MD_SUCCESS;
}

static int def_obj_name(char * out_name, int n, int d, int i){
  sprintf(out_name, "%s/%d_%d/file-%d", dir, n, d, i);
  return MD_SUCCESS;
}

static int create_dset(char * filename){
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL;
  int rc = 0;

  rc = parse_filename(filename, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", filename);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
	  fprintf(stderr, "Failed to lookup parent dir %s\n", dir_name);
	  return MD_ERROR_UNKNOWN;
  }

  rc = dfs_mkdir(dfs, parent, name, 0755);
  if (rc)
    fprintf(stderr, "dfs_mkdir() of %s Failed (%d)", filename, rc);

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);
  return rc;
}

static int rm_dset(char * filename){
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL;
  int rc;

  rc = parse_filename(filename, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", filename);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
	  fprintf(stderr, "Failed to lookup parent dir %s\n", dir_name);
	  return MD_ERROR_UNKNOWN;
  }

  rc = dfs_remove(dfs, parent, name, false, NULL);
  if (rc)
    fprintf(stderr, "dfs_remove() of %s Failed (%d)", filename, rc);

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return rc;
}

static int write_obj(char * dirname, char * filename, char * buf, size_t file_size)
{
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL, *obj = NULL;
  mode_t mode;
  int fd_oflag = 0;
  int rc;

  rc = parse_filename(filename, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", filename);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
    fprintf(stderr, "failed to lookup parent dir\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  mode = S_IFREG | 0644;
  fd_oflag |= O_CREAT | O_RDWR;

  rc = dfs_open(dfs, parent, name, mode, fd_oflag,
		objectClass, chunk_size, NULL, &obj);
  if (rc) {
    fprintf(stderr, "dfs_open() of %s Failed (%d)", filename, rc);
    return rc;
  }

  d_iov_t iov;
  d_sg_list_t sgl;

  /** set memory location */
  sgl.sg_nr = 1;
  sgl.sg_nr_out = 0;
  d_iov_set(&iov, (void *)buf, file_size);
  sgl.sg_iovs = &iov;

  rc = dfs_write(dfs, obj, sgl, 0);
  dfs_release(obj);
  if (rc) {
    fprintf(stderr, "dfs_write() failed (%d)", rc);
    return -1;
  }

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return MD_SUCCESS;
}

static int read_obj(char * dirname, char * filename, char * buf, size_t file_size)
{
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL, *obj = NULL;
  mode_t mode;
  int fd_oflag = 0;
  int rc;

  rc = parse_filename(filename, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", filename);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
    fprintf(stderr, "failed to lookup parent dir\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  mode = S_IFREG | 0644;
  fd_oflag |= O_RDWR;

  rc = dfs_open(dfs, parent, name, mode, fd_oflag,
		objectClass, chunk_size, NULL, &obj);
  if (rc) {
    fprintf(stderr, "dfs_open() of %s Failed (%d)", filename, rc);
    return rc;
  }

  d_iov_t iov;
  d_sg_list_t sgl;
  daos_size_t ret;

  /** set memory location */
  sgl.sg_nr = 1;
  sgl.sg_nr_out = 0;
  d_iov_set(&iov, (void *)buf, file_size);
  sgl.sg_iovs = &iov;

  rc = dfs_read(dfs, obj, sgl, 0, &ret);
  dfs_release(obj);
  if (rc) {
    fprintf(stderr, "dfs_write() failed (%d)", rc);
    return -1;
  }

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return MD_SUCCESS;
}

static int stat_obj(char * dirname, char * filename, size_t file_size){
  int rc;
  dfs_obj_t *parent = NULL;
  struct stat buf;

  parent = lookup_insert_dir(filename);
  if (parent == NULL) {
    fprintf(stderr, "failed to lookup filename to stat\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  rc = dfs_ostat(dfs, parent, &buf);
  DCHECK(rc, "dfs_stat() of Failed (%d)", rc);

  if ((size_t)buf.st_size != file_size){
    return MD_ERROR_UNKNOWN;
  }
  return MD_SUCCESS;
}

static int delete_obj(char * dirname, char * filename){
  char *name = NULL, *dir_name = NULL;
  dfs_obj_t *parent = NULL;
  int rc;

  rc = parse_filename(filename, &name, &dir_name);
  DCHECK(rc, "Failed to parse path %s", filename);

  parent = lookup_insert_dir(dir_name);
  if (parent == NULL) {
    fprintf(stderr, "failed to lookup parent dir\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  rc = dfs_remove(dfs, parent, name, false, NULL);
  if (rc)
    fprintf(stderr, "dfs_remove() of %s Failed (%d)", filename, rc);

  if (name)
	  free(name);
  if (dir_name)
	  free(dir_name);

  return rc;
}

struct md_plugin md_plugin_dfs = {
  "dfs",
  get_options,
  initialize,
  finalize,
  prepare_global,
  purge_global,

  def_dset_name,
  create_dset,
  rm_dset,

  def_obj_name,
  write_obj,
  read_obj,
  stat_obj,
  delete_obj
};
