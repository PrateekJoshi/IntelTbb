#include <iostream>
#include <tbb/tbb.h>

int main(int argc, char const *argv[])
{
    tbb::parallel_invoke(
        []() {
            std::cout << "Hello from task 1!" << std::endl;
        },
        []() {
            std::cout << "Hello from task 2!" << std::endl;
        }
    );
    return 0;
}
