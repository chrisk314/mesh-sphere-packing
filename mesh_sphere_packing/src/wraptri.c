/*****************************************************************************/
/*                                                                           */
/*  (tricall.c)                                                              */
/*                                                                           */
/*  Example program that demonstrates how to call Triangle.                  */
/*                                                                           */
/*  Accompanies Triangle Version 1.6                                         */
/*  July 19, 1996                                                            */
/*                                                                           */
/*  This file is placed in the public domain (but the file that it calls     */
/*  is still copyrighted!) by                                                */
/*  Jonathan Richard Shewchuk                                                */
/*  2360 Woolsey #H                                                          */
/*  Berkeley, California  94705-1927                                         */
/*  jrs@cs.berkeley.edu                                                      */
/*                                                                           */
/*****************************************************************************/

/* If SINGLE is defined when triangle.o is compiled, it should also be       */
/*   defined here.  If not, it should not be defined here.                   */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include <stdio.h>
#include <stdlib.h>
#include "triangle.h"

/*****************************************************************************/
/*                                                                           */
/*  report()   Print the input or output.                                    */
/*                                                                           */
/*****************************************************************************/

void report(io, markers, reporttriangles, reportneighbors, reportsegments,
            reportedges, reportnorms)
struct triangulateio *io;
int markers;
int reporttriangles;
int reportneighbors;
int reportsegments;
int reportedges;
int reportnorms;
{
  int i, j;

  for (i = 0; i < io->numberofpoints; i++) {
    printf("Point %4d:", i);
    for (j = 0; j < 2; j++) {
      printf("  %.6g", io->pointlist[i * 2 + j]);
    }
    if (io->numberofpointattributes > 0) {
      printf("   attributes");
    }
    for (j = 0; j < io->numberofpointattributes; j++) {
      printf("  %.6g",
             io->pointattributelist[i * io->numberofpointattributes + j]);
    }
    if (markers) {
      printf("   marker %d\n", io->pointmarkerlist[i]);
    } else {
      printf("\n");
    }
  }
  printf("\n");

  if (reporttriangles || reportneighbors) {
    for (i = 0; i < io->numberoftriangles; i++) {
      if (reporttriangles) {
        printf("Triangle %4d points:", i);
        for (j = 0; j < io->numberofcorners; j++) {
          printf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
        }
        if (io->numberoftriangleattributes > 0) {
          printf("   attributes");
        }
        for (j = 0; j < io->numberoftriangleattributes; j++) {
          printf("  %.6g", io->triangleattributelist[i *
                                         io->numberoftriangleattributes + j]);
        }
        printf("\n");
      }
      if (reportneighbors) {
        printf("Triangle %4d neighbors:", i);
        for (j = 0; j < 3; j++) {
          printf("  %4d", io->neighborlist[i * 3 + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportsegments) {
    for (i = 0; i < io->numberofsegments; i++) {
      printf("Segment %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->segmentlist[i * 2 + j]);
      }
      if (markers) {
        printf("   marker %d\n", io->segmentmarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportedges) {
    for (i = 0; i < io->numberofedges; i++) {
      printf("Edge %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->edgelist[i * 2 + j]);
      }
      if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
        for (j = 0; j < 2; j++) {
          printf("  %.6g", io->normlist[i * 2 + j]);
        }
      }
      if (markers) {
        printf("   marker %d\n", io->edgemarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }
}

/*****************************************************************************/
/*                                                                           */
/*  main()   Create and refine a mesh.                                       */
/*                                                                           */
/*****************************************************************************/

int wrap_tri(char* options, double* pointlist, double* pointattributes,\
    int numberofpoints, int numberofpointattributes, int* pointmarkerlist,\
    double* holelist, int numberofholes,
    double* regionlist, int numberofregions)
{
  struct triangulateio in, mid, out, vorout;

  /* Initialise input triangulateio pointers to NULL */
  in.pointlist = NULL; in.pointattributelist = NULL; in.pointmarkerlist = NULL;
  in.trianglelist = NULL; in.triangleattributelist = NULL; in.trianglearealist = NULL;
  in.segmentlist = NULL; in.segmentmarkerlist = NULL;
  in.holelist = NULL; in.regionlist = NULL;
  
  /* Define input points. */

  in.numberofpoints = numberofpoints;
  in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
  for (int i=0; i<2*in.numberofpoints; i++)
    in.pointlist[i] = pointlist[i];

  in.numberofpointattributes = numberofpointattributes;
  in.pointattributelist = (REAL *) malloc(in.numberofpoints *
                                          in.numberofpointattributes *
                                          sizeof(REAL));
  for (int i=0; i<in.numberofpoints*in.numberofpointattributes; i++)
    in.pointattributelist[i] = pointattributes[i];

  in.pointmarkerlist = (int *) malloc(in.numberofpoints * sizeof(int));
  for (int i=0; i<numberofpoints; i++)
    in.pointmarkerlist[i] = pointmarkerlist[i];

  in.numberofsegments = 0;

  in.numberofholes = numberofholes;
  if (in.numberofholes > 0) {
    in.holelist = (REAL *) malloc(2 * in.numberofholes * sizeof(REAL));
    for (int i=0; i<2*numberofholes; i++)
      in.holelist[i] = holelist[i];
  }

  in.numberofregions = numberofregions;
  if (in.numberofregions > 0){
    in.regionlist = (REAL *) malloc(in.numberofregions * 4 * sizeof(REAL));
    for (int i=0; i<4*numberofregions; i++)
      in.regionlist[i] = regionlist[i];
  }

  printf("Input point set:\n\n");
  report(&in, 1, 0, 0, 0, 0, 0);

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `mid' and a voronoi diagram in `vorout'.  */

  mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = (REAL *) NULL;
  mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = (REAL *) NULL;
  mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = (int *) NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = (int *) NULL;
  mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

  vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
  /* Needed only if -v switch used and number of attributes is not zero: */
  vorout.pointattributelist = (REAL *) NULL;
  vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
  vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

  /* Triangulate the points.  Switches are chosen to read and write a  */
  /*   PSLG (p), preserve the convex hull (c), number everything from  */
  /*   zero (z), assign a regional attribute to each element (A), and  */
  /*   produce an edge list (e), a Voronoi diagram (v), and a triangle */
  /*   neighbor list (n).                                              */

  triangulate(options, &in, &mid, &vorout);

  printf("Initial triangulation:\n\n");
  report(&mid, 1, 1, 1, 1, 1, 0);
  printf("Initial Voronoi diagram:\n\n");
  report(&vorout, 0, 0, 0, 0, 1, 1);

  /* Attach area constraints to the triangles in preparation for */
  /*   refining the triangulation.                               */

  /* Needed only if -r and -a switches used: */
  mid.trianglearealist = (REAL *) malloc(mid.numberoftriangles * sizeof(REAL));
  mid.trianglearealist[0] = 3.0;
  mid.trianglearealist[1] = 1.0;

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `out'.                                    */

  out.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of attributes is zero: */
  out.pointattributelist = (REAL *) NULL;
  out.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  out.triangleattributelist = (REAL *) NULL;

  /* Refine the triangulation according to the attached */
  /*   triangle area constraints.                       */

  triangulate("prazBP", &mid, &out, (struct triangulateio *) NULL);

  printf("Refined triangulation:\n\n");
  report(&out, 0, 1, 0, 0, 0, 0);

  /* Free all allocated arrays, including those allocated by Triangle. */

  free(in.pointlist);
  free(in.pointattributelist);
  if (in.pointmarkerlist != NULL) free(in.pointmarkerlist);
  if (in.holelist != NULL) free(in.holelist);
  if (in.regionlist != NULL) free(in.regionlist);
  free(mid.pointlist);
  free(mid.pointattributelist);
  free(mid.pointmarkerlist);
  free(mid.trianglelist);
  free(mid.triangleattributelist);
  free(mid.trianglearealist);
  free(mid.neighborlist);
  free(mid.segmentlist);
  free(mid.segmentmarkerlist);
  free(mid.edgelist);
  free(mid.edgemarkerlist);
  free(vorout.pointlist);
  free(vorout.pointattributelist);
  free(vorout.edgelist);
  free(vorout.normlist);
  free(out.pointlist);
  free(out.pointattributelist);
  free(out.trianglelist);
  free(out.triangleattributelist);

  return 0;
}
