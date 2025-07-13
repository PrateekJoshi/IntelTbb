// Pg 344
/*
Three-step checklist for deciding when work isolation is needed to ensure
correctness:
1. Is nested parallelism used (even indirectly, through third party
library calls)? If not, isolation is not needed; otherwise, go to the
next step.

2. Is it safe for a thread to reenter the outer level parallel tasks (as if
there was recursion)? Storing to a thread local value, re-acquiring
a mutex already acquired by this thread, or other resources
that should not be used by the same thread again can all cause
problems. If reentrance is safe, isolation is not needed; otherwise,
go to the next step.

3.Isolation is needed. Nested parallelism has to be called inside an
isolated region.

*/
#include <tbb/tbb.h>

void wrong_deadlock_code()
{
  tbb::spin_mutex m;
  const int P = oneapi::tbb::info::default_concurrency();
  tbb::parallel_for(0, P,
                    [&m](int)
                    {
                      const int N = 1000;
                      tbb::spin_mutex::scoped_lock l{m};
                      tbb::parallel_for(0, N,
                                        [](int j)
                                        {
                                          // Do some work
                                        });
                    });
}

void correct_fixed_code()
{
  tbb::spin_mutex m;
  const int P = oneapi::tbb::info::default_concurrency();
  tbb::parallel_for(0, P,
                    [&m](int)
                    {
                      const int N = 1000;
                      tbb::spin_mutex::scoped_lock l{m};
                      tbb::this_task_arena::isolate(
                          []()
                          {
                            tbb::parallel_for(0, N,
                                              [](int j)
                                              {
                                                // Do some work
                                              });
                          });
                    });
}

int main(int argc, char const *argv[])
{
  correct_fixed_code();
  return 0;
}
