#include <string.h>
#include <stdlib.h>
#include "mfh5.h"
#include "hdf5.h"
///#include "mpi.h"


int writeMeshMultiFlowH5(char *fname, double *nodes, int *cells, int *neighbours,\
    int *faces, int *facemarkers, int *faceadjcells, uint nNodes, uint nCells, uint nFaces)
{

  int i, j, start, n, nglobal;
  int *tmpInt;

  // HDF5 OUTPUT
  FILE *fp;
  hid_t file_id, plist_id;

  fp = fopen(fname, "w"); // Must create file before attempting to open with HDF5
  fclose(fp);

  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  file_id = H5Fopen(fname, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);
  H5Fclose(file_id);

  // nodes ----------------------------------------------------------------------------------------
  n = nNodes;
  nglobal = 3 * n;
  start = 0;

  MFArrayView(nodes, sizeof(double), "nodes", "/", fname, 3*n, start, nglobal, NULL);

  // cells ----------------------------------------------------------------------------------------
  n = 5 * nCells; // cell type + 4 cell node ids
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));

  for (i = 0; i < nCells; i++)
  {
    /* Cell type */
    tmpInt[5*i] = 6;

    /* CellNode IDs */
    for (j = 0; j < 4; j++)
      tmpInt[5*i+1+j] = cells[4*i+j];
  }

  MFArrayView(tmpInt, sizeof(int), "cells", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);

  // cellNodePtr ----------------------------------------------------------------------------------
  n = nCells + 1; // TODO why n + 2 not n + 1 ?
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));

  for (i = 0; i < n; i++)
    tmpInt[i] = 4*i;

  MFArrayView(tmpInt, sizeof(int), "cellNodePtr", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);

  // face -----------------------------------------------------------------------------------------
  n = 3 * nFaces;
  nglobal = n;
  start = 0;

  MFArrayView(faces, sizeof(int), "faces", "/", fname, n, start, nglobal, NULL);

  // faceNodePtr ----------------------------------------------------------------------------------
  n = nFaces + 1; // TODO why n + 2 not n + 1 ?
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));

  for (i = 0; i < n; i++)
    tmpInt[i] = 3*i;

  MFArrayView(tmpInt, sizeof(int), "faceNodePtr", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);

  // cellFaces ------------------------------------------------------------------------------------
  n = 4 * nCells;
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));
  
  int *faceCount = (int *) malloc(nCells * sizeof(int));
  memset(faceCount, 0, nCells * sizeof(int));

  for (i = 0; i < nFaces; i++){
    int c1 = faceadjcells[2*i];
    if (c1 > -1){
      tmpInt[4*c1+faceCount[c1]] = i;
      faceCount[c1]++;
    }
    int c2 = faceadjcells[2*i+1];
    if (c2 > -1){
      tmpInt[4*c2+faceCount[c2]] = i;
      faceCount[c2]++;
    }
  }

  MFArrayView(tmpInt, sizeof(int), "cellFaces", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);
  free(faceCount);

  // cellFacePtr ----------------------------------------------------------------------------------
  n = nCells + 1;
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));

  for (i = 0; i < n; i++)
    tmpInt[i] = 4*i;

  MFArrayView(tmpInt, sizeof(int), "cellFacePtr", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);

  // boundaryType ---------------------------------------------------------------------------------
  n = nFaces;
  nglobal = n;
  start = 0;

  MFArrayView(facemarkers, sizeof(int), "boundaryType", "/", fname, n, start, nglobal, NULL);

  // cellNeighbours -------------------------------------------------------------------------------
  n = 4 * nCells;
  nglobal = n;
  start = 0;

  tmpInt = (int *) malloc(n * sizeof(int));
  int *tmpNeighbourPtr = (int *) malloc((nCells + 1) * sizeof(int));
  
  int nbrCount = 0;
  for (i = 0; i < n; i++){
    if (i % 4 == 0)
      tmpNeighbourPtr[i/4] = nbrCount;
    if (neighbours[i] > -1)
      tmpInt[nbrCount++] = neighbours[i];
  }
  tmpNeighbourPtr[nCells] = nbrCount;

  MFArrayView(tmpInt, sizeof(int), "cellNeighbours", "/", fname, nbrCount, start, nbrCount, NULL);
  free(tmpInt);

  // cellNeighbourPtr -----------------------------------------------------------------------------
  n = nCells + 1;
  nglobal = n;
  start = 0;

  MFArrayView(tmpNeighbourPtr, sizeof(int), "cellNeighbourPtr", "/", fname, n, start, nglobal, NULL);
  free(tmpNeighbourPtr);

  /* Mesh header data -----------------------------------------------------------------------------
   *
   * [0]:    (int) number of nodes
   * [1]:    (int) number of cells
   * [2]:    (int) number of faces
   * [3]:    (int) cell type: 4 = tet, 5 = hex, 8 = poly
   * [4]:    (int) length of cellNode array
   * [5]:    (int) length of faceNode array
   * [6]:    (int) length of cellFace array
   *
   */
  tmpInt = (int *) malloc(8 * sizeof(int));

  tmpInt[0] = nNodes;
  tmpInt[1] = nCells;
  tmpInt[2] = nFaces;

  tmpInt[3] = 8;    // Mixed mesh (generic for unstructured)
  tmpInt[4] = 4 * nCells;
  tmpInt[5] = 3 * nFaces;
  tmpInt[6] = 4 * nCells;
  tmpInt[7] = nbrCount;

  nglobal = 8;
  start = 0;
  n = 8;

  MFArrayView(tmpInt, sizeof(int), "meshData", "/", fname, n, start, nglobal, NULL);
  free(tmpInt);

  return 0;
}


int MFArrayView(void *array, size_t type, char *dataset, char *group, char *filename, int n,
    int start, int nglobal, size_t *idx)
{
  char path[500];
  hid_t file_id, dset_id, mem_type_id; /* file and dataset identifiers */
  hid_t filespace, memspace; /* file and memory dataspace identifiers */
  hsize_t dimsf[1]; /* dataset dimensions */
  hsize_t count[1]; /* hyperslab selection parameters */
  hsize_t offset[1];
  hid_t plist_id; /* property list identifier */
  herr_t status;

  //MPI_Comm comm = MPI_COMM_WORLD;
  //MPI_Info info = MPI_INFO_NULL;

  /* Define type to print */
  if (type == sizeof(int))
    mem_type_id = H5T_NATIVE_INT;
  else if (type == sizeof(double))
    mem_type_id = H5T_NATIVE_DOUBLE;

  /* Set up file access property list with parallel I/O access */
  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  /* Open file collectively and release property list identifier */
  file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  /* Create the dataspace for the dataset */
  dimsf[0] = nglobal;
  count[0] = n;
  filespace = H5Screate_simple(1, dimsf, NULL);
  memspace = H5Screate_simple(1, count, NULL);

  sprintf(path, "%s/%s", group, dataset);
  if (!H5Lexists(file_id, path, H5P_DEFAULT))
  {
    dset_id = H5Dcreate(file_id, path, mem_type_id, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
  }
  else
  {
    dset_id = H5Dopen(file_id, path, H5P_DEFAULT);
  }

  filespace = H5Dget_space(dset_id);

  if (idx == NULL)
  {
    offset[0] = start;
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
  }
  else
  {
    status = H5Sselect_elements(filespace, H5S_SELECT_SET, (size_t) n, idx);
  }

  /* Create property list for collective dataset write */
  plist_id = H5Pcreate(H5P_DATASET_XFER);

  status = H5Dwrite(dset_id, mem_type_id, memspace, filespace, plist_id, array);

  /* Close/release resources */
  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Fclose(file_id);

  return 0;
}


int MFArrayLoad(void **array, size_t type, char *dataset, char *group, char *filename, int n,
    int start, int nglobal, size_t *idx)
{
  char path[500];
  hid_t file_id, dset_id, mem_type_id; /* file and dataset identifiers */
  hid_t filespace, memspace; /* file and memory dataspace identifiers */
  hsize_t dimsf[1]; /* dataset dimensions */
  hsize_t count[1]; /* hyperslab selection parameters */
  hsize_t offset[1];
  hid_t plist_id; /* property list identifier */
  herr_t status;

  //MPI_Comm comm = MPI_COMM_WORLD;
  //MPI_Info info = MPI_INFO_NULL;

  /* Define type to print */
  if (type == sizeof(int))
    mem_type_id = H5T_NATIVE_INT;
  else if (type == sizeof(double))
    mem_type_id = H5T_NATIVE_DOUBLE;

  /* Set up file access property list with parallel I/O access */
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  //H5Pset_fapl_mpio(plist_id, comm, info);

  /* Open file collectively and release property list identifier */
  file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  /* Open dataset */
  sprintf(path, "%s/%s", group, dataset);
  dset_id = H5Dopen(file_id, path, H5P_DEFAULT);

  /* Create the dataspace for the dataset */
  dimsf[0] = (size_t) nglobal;
  count[0] = (size_t) n;
  filespace = H5Screate_simple(1, dimsf, NULL);
  memspace = H5Screate_simple(1, count, NULL);
  filespace = H5Dget_space(dset_id);

  if (idx == NULL)
  {
    offset[0] = (size_t) start;
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
  }
  else
  {
    status = H5Sselect_elements(filespace, H5S_SELECT_SET, (size_t) n, idx);
  }

  /* Create property list for collective dataset write */
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  status = H5Dread(dset_id, mem_type_id, memspace, filespace, plist_id, *array);

  /* Close/release resources */
  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Fclose(file_id);

  return 0;
}
