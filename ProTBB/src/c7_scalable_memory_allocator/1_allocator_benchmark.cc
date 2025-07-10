/*
(conanenv) prateek@prateek:~/Workspace/IntelTbb/build/src/c7_scalable_memory_allocator$ ./1_allocator_benchmark 
--- Multithreaded Allocator Benchmark ---
Allocator: std::allocator
Threads: 8
Allocations per thread: 200000
Allocation size: 64 bytes
Total allocations: 1600000

Benchmark completed in: 0.0603033 seconds
---------------------------------------
(conanenv) prateek@prateek:~/Workspace/IntelTbb/build/src/c7_scalable_memory_allocator$ ./2_allocator_benchmark 
--- Multithreaded Allocator Benchmark ---
Allocator: tbb::scalable_allocator
Threads: 8
Allocations per thread: 200000
Allocation size: 64 bytes
Total allocations: 1600000

Benchmark completed in: 0.0563807 seconds
---------------------------------------
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric> // For std::iota
#include <string>  // For allocation content

// Define the allocator type based on a preprocessor macro
// This allows compiling the same code with different allocators
#ifdef USE_TBB_ALLOCATOR
    #include <tbb/scalable_allocator.h>
    template<typename T>
    using MyAllocator = tbb::scalable_allocator<T>;
    const char* ALLOCATOR_NAME = "tbb::scalable_allocator";
#else
    #include <memory> // For std::allocator
    template<typename T>
    using MyAllocator = std::allocator<T>;
    const char* ALLOCATOR_NAME = "std::allocator";
#endif

// --- Configuration ---
const int NUM_THREADS = 8; // Number of threads to use
const int ALLOCATIONS_PER_THREAD = 200000; // Number of allocations/deallocations per thread
const size_t ALLOCATION_SIZE = 64; // Size of each allocated object in bytes

// A simple struct to represent the allocated data
// Using char array to control exact size
struct Data {
    char bytes[ALLOCATION_SIZE];

    // Constructor to initialize data (optional, but good for realistic scenarios)
    Data() {
        std::iota(std::begin(bytes), std::end(bytes), 0); // Fill with some pattern
    }
};

// Function executed by each thread
void run_alloc_dealloc_benchmark() {
    // Use a vector with the chosen allocator
    // This will allocate 'Data' objects using MyAllocator
    std::vector<Data, MyAllocator<Data>> data_vec;
    data_vec.reserve(ALLOCATIONS_PER_THREAD); // Pre-allocate vector capacity to avoid reallocations affecting benchmark

    // Phase 1: Allocate objects
    for (int i = 0; i < ALLOCATIONS_PER_THREAD; ++i) {
        data_vec.emplace_back(); // Allocate and construct a Data object
    }

    // Phase 2: Deallocate objects (by clearing the vector)
    // The destructor of std::vector will call the allocator's deallocate method for each element
    data_vec.clear();
    data_vec.shrink_to_fit(); // Release memory back to the allocator
}

int main() {
    std::cout << "--- Multithreaded Allocator Benchmark ---" << std::endl;
    std::cout << "Allocator: " << ALLOCATOR_NAME << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Allocations per thread: " << ALLOCATIONS_PER_THREAD << std::endl;
    std::cout << "Allocation size: " << ALLOCATION_SIZE << " bytes" << std::endl;
    std::cout << "Total allocations: " << (long long)NUM_THREADS * ALLOCATIONS_PER_THREAD << std::endl;
    std::cout << std::endl;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(run_alloc_dealloc_benchmark);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Benchmark completed in: " << elapsed_time.count() << " seconds" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    return 0;
}
