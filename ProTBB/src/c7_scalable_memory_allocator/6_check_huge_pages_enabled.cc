// Pg 228
#include <tbb/scalable_allocator.h>
#include <iostream>

int main() {
    int mode = scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, 1);
    if (mode == TBBMALLOC_NO_EFFECT) {
        std::cout << "Huge pages are disabled in TBB\n";
    }
    else if( mode == TBBMALLOC_OK )
    {
        std::cout << "Huge pages are enabled in TBB\n";
    }
    else 
    {
        std::cout << "Error mode :"<<mode<<"\n";
    }
    return 0;
}