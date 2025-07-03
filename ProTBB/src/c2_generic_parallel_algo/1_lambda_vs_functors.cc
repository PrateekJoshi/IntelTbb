// Pg - 36
#include <tbb/parallel_for.h>
#include <vector>
#include <iostream>

void lambda()
{
    std::vector<int> data(10, 0);

    tbb::parallel_for(0, static_cast<int>(data.size()), 1, 
        [&data](int i) {
            data[i] += 1;
        });

    for (int v : data) std::cout << v << " ";
}

void functors()
{
    std::vector<int> data(10, 0);

    struct Increment {
        std::vector<int>& data;
        Increment(std::vector<int>& d) : data(d) {}
        void operator()(int i) const {
            data[i] += 1;
        }
    };

    tbb::parallel_for(0, static_cast<int>(data.size()), 1, Increment(data));

    for (int v : data) std::cout << v << " ";
}

int main() {
    lambda();
    std::cout << std::endl;
    functors();
    return 0;
}