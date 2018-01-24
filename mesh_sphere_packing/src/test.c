#include <assert.h>
#include <stdio.h>
#include "wraptri.h"

int main () {
  assert(wrap_tri() == 0);
  printf("Tests succesfully completed.\n");
  return 0;  
}
