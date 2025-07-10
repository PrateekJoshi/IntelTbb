// Pg 219
#include <stdio.h>
#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

#include <stdio.h>
#include "tbb/tbb.h"

using namespace tbb;

const int N = 1000000;

// don't forget ulimit –s unlimited on Linux, or STACK:10000000 on Windows
// otherwise this will fail to run

int main() {
  double *a[N];

  parallel_for( 0, N-1, [&](int i) { a[i] = new double; } );
  parallel_for( 0, N-1, [&](int i) { delete a[i];       } );

  return 0;
}