# Pro TBB 

## Preface 

### 🧠 Nested Parallelism: TBB vs. OpenMP

#### ✅ What is Nested Parallelism?
> **Nested parallelism** occurs when a parallel region spawns another parallel region inside it.

---

#### 🔄 TBB: Natural Nested Tasking

- **Work-Stealing Scheduler**  
  Efficiently balances nested tasks across threads without manual intervention.

- **Lightweight Tasks**  
  TBB tasks are much lighter than OS threads, making deep nesting scalable.

- **Composable Algorithms**  
  Nested parallel calls "just work"—you can parallelize in one function and call it from another that's also parallel.

- **Scales Gracefully**  
  No need to tune thread counts manually. TBB dynamically adapts to hardware and workload.

---

#### ⚠️ OpenMP: Limited Nesting Efficiency

- **Thread Oversubscription Risk**  
  Each nested parallel region may create a new thread team → leads to contention and slowdown.

- **Needs Manual Control**  
  Must enable nested parallelism with `omp_set_nested(1)` and configure thread limits.

- **Less Composability**  
  Nested parallel sections may require careful coordination and resource management.

- **Heavier Overhead**  
  Threads in OpenMP are typically OS threads—more costly to manage compared to TBB tasks.

---

#### 📊 Side-by-Side Comparison

| Feature                | **Intel TBB**                | **OpenMP**                     |
|------------------------|------------------------------|-------------------------------|
| Scheduling             | Dynamic work-stealing         | Static or user-defined        |
| Nested Region Support  | Native, composable            | Must enable manually          |
| Thread Management      | Lightweight tasks             | OS-level threads              |
| Load Balancing         | Automatic                     | Manual tuning needed          |
| Performance on Nested Tasks | Scales well               | May degrade                   |

---

#### 💡 Verdict
> If you're building modern, modular C++ apps with dynamic workloads, **TBB’s composable nested parallelism** makes it more powerful and flexible than OpenMP in complex scenarios.

### Amdahl’s Law vs Gustafson’s Law

| **Feature**               | **Amdahl’s Law**                                                                 | **Gustafson’s Law**                                                            |
|---------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| **Originator**            | Gene Amdahl (1967)                                                               | John L. Gustafson (1988)                                                       |
| **Focus**                 | Limits of speedup due to the serial portion of a task                            | Scalability of parallel systems with increasing problem size                   |
| **Assumption**            | Fixed problem size                                                               | Problem size scales with the number of processors                              |
| **Formula**               | `Speedup = 1 / [(1 - P) + (P / N)]`                                              | `Speedup = N - (1 - P) × (N - 1)`                                              |
| **Interpretation**        | Serial bottlenecks limit speedup, even with more processors                      | Efficient parallel performance with increased workload                         |
| **Viewpoint**             | Pessimistic about the benefits of parallelism                                    | Optimistic about system scalability                                            |
| **Use Case**              | Evaluating limits for existing workloads                                         | Designing scalable systems for larger workloads                                |
| **Limitation**            | Doesn’t scale problem size; ignores memory or communication overhead             | Assumes perfect scaling; may not reflect practical system limitations          |

### Locality of Reference

Accessing memory is costly the first time, but faster afterward due to caching. Like short-term memory, caches remember recent data. When loops access memory contiguously (good spatial locality) and reuse data soon (good temporal locality), performance improves.

In C/C++, arrays are stored in row-major order, so elements like C[i][2] and C[i][3] are adjacent in memory, while C[2][j] and C[3][j] are not.

❌ Poor Locality: Inefficient Loop Order

```cpp
for (int i = 0; i < N; ++i)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < N; ++k)
      C[i][j] += A[i][k] * B[k][j];  // Poor spatial locality for B
```

- Accesses B column-wise in a row-major system ⇒ cache misses.

✅ Improved Locality: Swapped Loop Order

```cpp
for (int i = 0; i < N; ++i)
  for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
      C[i][j] += A[i][k] * B[k][j];  // Better cache utilization
```

- Now both A and B are accessed more contiguously ⇒ better cache hits.

### Data ALignment vs Data Sharing vs False Sharing

| Feature               | Data Alignment                                                                 | Data Sharing                                                                                     | False Sharing                                                                                   |
|-----------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Definition**         | Arranging data in memory according to hardware alignment boundaries           | Allowing multiple processes or threads to access the same data                                   | Multiple threads access different variables that reside on the same cache line                  |
| **Goal**               | Improve memory access speed and enable vectorization                          | Enable collaboration and reduce data duplication                                                 | Often unintentional; leads to performance degradation                                           |
| **Performance Impact** | ✅ Positive: improves cache efficiency and SIMD performance                   | ⚖️ Neutral to positive: depends on synchronization and access pattern                            | ❌ Negative: increases memory latency due to cache line bouncing                                |
| **Typical Use Case**   | Structuring arrays/structs for SIMD operations                                | Shared memory in MPI/OpenMP, collaborative tasks                                                  | Multithreaded programs with adjacent variables accessed by separate threads                     |
| **Detection**          | Compiler warnings, profiling tools                                            | File system permissions or shared memory inspection                                               | Hardware counters (e.g. HITM events), Intel VTune, LIKWID                                       |
| **Mitigation**         | Use `__attribute__((aligned(N)))`, `memalign()` or compiler flags             | Apply appropriate access control mechanisms (ACLs, POSIX groups)                                 | Add padding, align variables to cache line size (e.g., 64 bytes), avoid shared writes           |
| **Example Scenario**   | Aligning `double` arrays to 64-byte boundaries for AVX-512                    | Users accessing shared simulation input files on an HPC cluster                                  | Two counters for different threads allocated on the same cache line                             |


### False sharing 

- Simple C++ example that demonstrates false sharing and how to fix it using padding to avoid multiple threads writing to the same cache line.

🚫 Example with False Sharing

```cpp
#include <iostream>
#include <thread>
#include <vector>

constexpr int NUM_THREADS = 4;
constexpr long NUM_ITER = 100000000;

struct SharedData {
    int counter[NUM_THREADS]; // All counters are adjacent in memory
};

void increment(SharedData* data, int index) {
    for (long i = 0; i < NUM_ITER; ++i) {
        data->counter[index]++;
    }
}

int main() {
    SharedData data{};
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(increment, &data, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        std::cout << "Counter[" << i << "] = " << data.counter[i] << "\n";
    }

    return 0;
}
```

In this version, all `counter[i]` values are likely to reside on the same cache line, causing false sharing when multiple threads update them simultaneously.

---

✅ Fixed Version with Padding

```cpp
#include <iostream>
#include <thread>
#include <vector>

constexpr int NUM_THREADS = 4;
constexpr long NUM_ITER = 100000000;
constexpr int CACHE_LINE_SIZE = 64;

struct alignas(CACHE_LINE_SIZE) PaddedCounter {
    int value;
    char padding[CACHE_LINE_SIZE - sizeof(int)];
};

struct SharedData {
    PaddedCounter counter[NUM_THREADS];
};

void increment(SharedData* data, int index) {
    for (long i = 0; i < NUM_ITER; ++i) {
        data->counter[index].value++;
    }
}

int main() {
    SharedData data{};
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(increment, &data, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        std::cout << "Counter[" << i << "] = " << data.counter[i].value << "\n";
    }

    return 0;
}
```

By aligning and padding each `counter` to a separate cache line, we eliminate false sharing and significantly improve performance in multithreaded environments.

### 🧠 Understanding the Power of Threads + Vectorization

Modern CPUs can do two big things to speed up programs:

#### 1. **Multithreading (using all CPU cores)**
- Imagine your processor has **32 cores**—like having 32 workers doing tasks at the same time.
- A library like **Intel TBB** can help your program use **all these cores together**.
- This could give you **up to 32× speedup**, if everything scales perfectly.

#### 2. **Vectorization (each core works faster)**
- Each core also has tools (like **AVX instructions**) to process **multiple numbers at once**.
- This is like one worker doing **4 things at once**.
- So even with just one core, you could get **up to 4× faster math**.

#### 🚀 Put Them Together: Multiplied Performance
- If you combine both techniques:
  - **32 cores** × **4 numbers at once** = **theoretically 256× faster** 💥
- That's why performance-focused developers use **both**: threads **and** vector instructions.

> It’s like giving each worker a power tool—and then hiring 31 more of them. ⚡🛠️


## References 

- [ All of the code examples used in this book are available ](https://github.com/Apress/pro-TBB)