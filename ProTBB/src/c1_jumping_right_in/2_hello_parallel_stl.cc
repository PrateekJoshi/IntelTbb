#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>

int main(int argc, char const *argv[])
{
    std::vector<std::string> v = {"Hello","Parallel","STL"};
    std::for_each(std::execution::par, v.begin(), v.end(), [](const std::string& s) {
        std::cout << s << " "<<std::endl;
    });
    return 0;
}
