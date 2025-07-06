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

---

## Chapter 2 : Generic Parallel Algorithms

### The Generic Algorithms in the Threading Building Blocks library

| 🗂️ **Category**            | 🧩 **Generic Algorithm**         | 📝 **Brief Description**                                                                 | 🧪 **Syntax**                                                                 |
|---------------------------|----------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Loop Parallelism**      | `parallel_for`                   | Executes loop iterations in parallel over a range.                                       | `parallel_for(range, body);`                                                 |
|                           | `parallel_reduce`                | Parallel loop with a reduction operation (e.g., sum).                                    | `parallel_reduce(range, identity, func, reduction);`                         |
|                           | `parallel_deterministic_reduce` | Like `parallel_reduce`, but ensures deterministic results regardless of thread count.    | `parallel_deterministic_reduce(range, identity, func, reduction);`          |
|                           | `parallel_scan`                  | Computes prefix sums (inclusive/exclusive scan).                                         | `parallel_scan(range, body);`                                                |
| **Function Invocation**   | `parallel_invoke`                | Executes multiple functions concurrently.                                                | `parallel_invoke(func1, func2, ..., funcN);`                                 |
| **Stream Processing**     | `parallel_do`                    | Processes a stream of tasks with dynamic work addition.                                  | `parallel_do(input_range, body);`                                            |
|                           | `parallel_for_each`              | Applies a function to each element in a range (parallel version of `std::for_each`).     | `parallel_for_each(begin, end, func);`                                       |
| **Sorting**               | `parallel_sort`                  | Sorts elements in parallel using comparison-based sorting.                               | `parallel_sort(begin, end);`                                                 |
| **Pipeline Processing**   | `pipeline`                       | Constructs a linear pipeline with custom serial or parallel filters.                     | `pipeline p; p.add_filter(...); p.run(token_count);`                         |
|                           | `parallel_pipeline`              | Higher-level composable pipeline using chained filter stages via operator `&`.           | `parallel_pipeline(token_count, stage1 & stage2 & ...);`                     |


### LAMBDA EXPRESSIONS –VS- USER-DEFINED CLASSES

#### ✅ Using Lambda Expression

```cpp
#include <tbb/parallel_for.h>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> data(10, 0);

    tbb::parallel_for(0, static_cast<int>(data.size()), 1, 
        [&data](int i) {
            data[i] += 1;
        });

    for (int v : data) std::cout << v << " ";
    return 0;
}
```
#### 🧱 Using Functor (Function Object)

```cpp
#include <tbb/parallel_for.h>
#include <vector>
#include <iostream>

struct IncrementFunctor {
    std::vector<int>& data;

    IncrementFunctor(std::vector<int>& d) : data(d) {}

    void operator()(int i) const {
        data[i] += 1;
    }
};

int main() {
    std::vector<int> data(10, 0);

    tbb::parallel_for(0, static_cast<int>(data.size()), 1, IncrementFunctor(data));

    for (int v : data) std::cout << v << " ";
    return 0;
}
```

#### 📊 Summary: Lambda vs Functor in `tbb::parallel_for`

| Feature            | Lambda Expression                           | Functor                                       |
|--------------------|----------------------------------------------|-----------------------------------------------|
| **Syntax Style**   | Short, inline                                | Separate class with `operator()`              |
| **Use Case**       | Ideal for simple, local logic                | Better for reusable or complex logic          |
| **State Handling** | Captures external variables (by ref/value)   | Encapsulates internal state as members        |
| **Reusability**    | Limited to the scope it's defined in         | Can be reused across codebases and files      |
| **Readability**    | Very clean for small snippets                | More descriptive for structured operations     |
| **Testability**    | Harder to test in isolation                  | Easy to write unit tests for the functor      |


### Intel TBB Lazy initilization and warmup

- Intel oneTBB (Threading Building Blocks) lazily initializes its worker threads—meaning they’re only created when needed. This can cause a delay the first time you run a parallel algorithm.

- `warmupTBB()` below preemptively starts those threads so that subsequent parallel operations don’t suffer from that startup cost.

```cpp
void warmupTBB() {
  tbb::parallel_for(0, tbb::info::default_concurrency(), 
    [=](int) {
      tbb::tick_count t0 = tbb::tick_count::now();
      while ((tbb::tick_count::now() - t0).seconds() < 0.01);
    }
  );
}
```

🧠 `warmupTBB()` Function Explained

| 🔢 Line or Component                                 | 📝 Explanation |
|------------------------------------------------------|----------------|
| `static void warmupTBB()`                            | Defines a static helper function to warm up the TBB thread pool. |
| `tbb::parallel_for(...)`                             | Runs a loop in parallel across available threads. |
| `0, tbb::info::default_concurrency()`                | Loop from 0 to the number of hardware threads (cores). |
| `[=](int)`                                           | Lambda capturing all variables by value (loop index unused). |
| `tbb::tick_count t0 = tbb::tick_count::now();`       | Stores the current timestamp as a starting point. |
| `while ((now - t0).seconds() < 0.01);`               | Busy-waits for 10 milliseconds to keep the thread active. |


## Chapter 3 : Flow Graphs

### 🔗 Intel TBB `join_node` Types Explained

A `join_node` in Intel TBB's Flow Graph API synchronizes multiple inputs. It waits for messages on all its input ports, then combines them into a tuple and sends that tuple to the next node.

---

#### 🧩 Types of `join_node`

| Type            | Description                                                                 | Use Case                                  |
|-----------------|-----------------------------------------------------------------------------|-------------------------------------------|
| `queueing`      | Waits for one message on each port, in the order they arrive.               | Simple synchronization.                   |
| `reserving`     | Reserves messages until all inputs are available, then consumes them.       | Prevents message loss or premature use.   |
| `key_matching`  | Matches messages based on a shared key (e.g., ID).                          | When inputs must be matched by key.       |

---

#### 🧪 Examples

##### 1. `queueing` Join Node

```cpp
#include <tbb/flow_graph.h>
#include <iostream>

using namespace tbb::flow;

int main() {
    graph g;

    function_node<int> a(g, unlimited, [](int x) { return x; });
    function_node<float> b(g, unlimited, [](float y) { return y; });

    join_node<tuple<int, float>, queueing> j(g);

    function_node<tuple<int, float>> c(g, unlimited, [](const tuple<int, float>& t) {
        std::cout << "Got: " << get<0>(t) << " and " << get<1>(t) << std::endl;
    });

    make_edge(a, input_port<0>(j));
    make_edge(b, input_port<1>(j));
    make_edge(j, c);

    a.try_put(42);
    b.try_put(3.14f);
    g.wait_for_all();
}
```

🧠 __Explanation__: Waits for one int and one float, then sends the tuple to node c.

##### 2. `key_matching` Join Node

```cpp
join_node<tuple<int, float>, reserving> j(g);
```

🧠 __Explanation__: Reserves messages on each port and only removes them when all are available. Prevents consuming one input if the others aren’t ready.

##### 3. `reserving` Join Node

```cpp
#include <string>

struct Data {
    int key;
    std::string value;
};

auto key_func = [](const Data& d) { return d.key; };

join_node<tuple<Data, Data>, key_matching<int>> j(g, key_func, key_func);
```

🧠 __Explanation__ : Matches two Data objects with the same key. Useful when inputs arrive out of order but must be paired by ID.

#### ✅ Summary

- Use `queueing` for simple, ordered synchronization.

- Use `reserving` to avoid premature message consumption.

- Use `key_matching` when inputs must be matched by a shared key.

### 📊 Data Flow Graph vs Dependency Graph in Intel oneAPI (oneTBB)

This table compares the two primary graph paradigms in Intel oneAPI's Threading Building Blocks (TBB) Flow Graph API.

| Feature / Aspect              | **Data Flow Graph**                                                                 | **Dependency Graph**                                                                 |
|------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Definition**               | A graph where **data values** are passed between nodes.                            | A graph where **task completion signals** (not data) are passed between nodes.       |
| **Message Type**             | Arbitrary data types (e.g., `int`, `std::string`, custom structs).                 | Always uses `continue_msg` to signal task completion.                                |
| **Node Types Used**          | `function_node`, `input_node`, `source_node`, `join_node`, etc.                    | Primarily `continue_node`, sometimes `broadcast_node`, `join_node`.                  |
| **Purpose**                  | Models **data-driven execution**: a node runs when it receives data.               | Models **control dependencies**: a node runs when all its predecessors complete.     |
| **Data Transfer**            | ✅ Yes — actual data is passed from node to node.                                   | ❌ No — only a signal (`continue_msg`) is passed; data is accessed via shared state. |
| **Execution Trigger**        | A node executes when it receives a message with data.                              | A node executes when it receives the required number of `continue_msg`s.             |
| **Use Case**                 | Streaming pipelines, transformations, filtering, etc.                              | Task orchestration, dependency resolution, barrier-like behavior.                    |
| **Example Node**             | `function_node<int>` receives and processes integers.                              | `continue_node` triggers execution after predecessors complete.                      |
| **Parallelism Granularity** | Fine-grained, task-per-message.                                                    | Coarser-grained, task-per-dependency.                                                |
| **Flexibility**              | High — supports complex data flows and transformations.                            | Simpler — focused on expressing execution order.                                     |
| **Example Scenario**         | Image processing pipeline: load → filter → compress.                              | Task A and B must finish before C starts.                                            |

---

#### 🧠 Summary

- Use a **Data Flow Graph** when your computation is **driven by data** and you want to pass values between stages.
- Use a **Dependency Graph** when your computation is **driven by task completion** and you just need to enforce execution order.



## References 

- [ All of the code examples used in this book are available ](https://github.com/Apress/pro-TBB)