#include <stdio.h>                   // Standard I/O library (used here for completeness)
#include <tbb/scalable_allocator.h>  // Provides scalable_malloc and scalable_free for custom allocation
#include <tbb/tbb.h>                 // Includes the core Intel TBB functionalities

void* operator new(size_t size, const std::nothrow_t&) noexcept
{
  if (size == 0) size = 1;                // Avoid zero-byte allocation, which is undefined behavior
  if (void* ptr = scalable_malloc(size))  // Try to allocate memory using TBB's scalable allocator
    return ptr;                           // Return pointer if successful
  return NULL;                            // Return null on failure (nothrow variant doesn't throw)
}

void* operator new[](size_t size, const std::nothrow_t&) noexcept
{
  return operator new(size, std::nothrow);  // Reuse the scalar version for array allocation
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept
{
  if (ptr != 0) scalable_free(ptr);  // Free memory using scalable_free, if not null
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept
{
  operator delete(ptr, std::nothrow);  // Reuse scalar delete logic for array delete
}

int main()
{
  const int N = 1000000;  // Total number of allocations to perform
  double* a[N];           // Array of N double pointers on the stack

  // Parallel allocation: each thread allocates one double using custom new
  tbb::parallel_for(0, N - 1,
                    [&](int i)
                    {
                      a[i] = new double;  // Allocates one double, using scalable_malloc due to overload
                    });

  // Parallel deallocation: each thread frees the allocated double
  tbb::parallel_for(0, N - 1,
                    [&](int i)
                    {
                      delete a[i];  // Deletes the allocated double, using scalable_free
                    });

  return 0;
}
