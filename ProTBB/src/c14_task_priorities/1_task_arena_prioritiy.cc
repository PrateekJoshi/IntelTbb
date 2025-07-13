// Pg 378
#include <oneapi/tbb.h>
#include <iostream>
#include <thread>

void high_priority_work() {
    std::cout << "[High Priority] Task running on thread " << std::this_thread::get_id() << "\n";
}

void low_priority_work() {
    std::cout << "[Low Priority] Task running on thread " << std::this_thread::get_id() << "\n";
}

int main() {
    // Create two arenas with different concurrency levels
    oneapi::tbb::task_arena high_priority_arena(4); // Simulate high priority
    oneapi::tbb::task_arena low_priority_arena(2);  // Simulate low priority

    // Launch high-priority tasks
    high_priority_arena.execute([] {
        oneapi::tbb::task_group tg;
        for (int i = 0; i < 4; ++i) {
            tg.run(high_priority_work);
        }
        tg.wait();
    });

    // Launch low-priority tasks
    low_priority_arena.execute([] {
        oneapi::tbb::task_group tg;
        for (int i = 0; i < 4; ++i) {
            tg.run(low_priority_work);
        }
        tg.wait();
    });

    return 0;
}
