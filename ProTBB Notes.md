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