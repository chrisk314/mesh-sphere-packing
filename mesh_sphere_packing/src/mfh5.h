#ifndef INCLUDE_MFH5_H_
#define INCLUDE_MFH5_H_

#include <sys/types.h>

typedef unsigned int uint;

int writeMeshMultiFlowH5(char *fname, double *nodes, int *cells, int *neighbours,
    int *faces, int *facemarkers, int *faceadjcells, uint nNodes, uint nCells, uint nFaces);

int MFArrayView(void *array, size_t type, char *dataset, char *group, char *filename, int n,
      int start, int nglobal, size_t *idx);

int MFArrayLoad(void **array, size_t type, char *dataset, char *group, char *filename, int n,
      int start, int nglobal, size_t *idx);

#endif /* INCLUDE_MFH5_H_ */
