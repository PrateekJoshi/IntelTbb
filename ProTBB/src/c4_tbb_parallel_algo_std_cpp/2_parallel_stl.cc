// Pg 129 
#include <iostream>
#include <vector>
#include <numeric>      // for std::reduce, std::transform_reduce
#include <algorithm>    // for std::transform
#include <execution>    // for std::execution::par_unseq

// 🧠 Use case: Apply a transformation (e.g., squaring) in parallel and vectorized fashion.
void std_transform()
{
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(input.size());

    std::transform(std::execution::par_unseq, input.begin(), input.end(), output.begin(),
                   [](int x) { return x * x; });

    std::cout << "Transformed (squared): ";
    for (int val : output) std::cout << val << " ";
    std::cout << "\n";
}

// 🧠 Use case: Fast summation of large vectors using parallel reduction.
void std_reduce()
{
    std::vector<int> input = {1, 2, 3, 4, 5};

    // 🧠 Use case: Compute the sum of elements in parallel.
    int sum = std::reduce(std::execution::par_unseq, input.begin(), input.end(), 0);

    std::cout << "Sum: " << sum << "\n";
}

void transform_reduce()
{
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30, 40, 50};

    int dot_product = std::transform_reduce(
        std::execution::par_unseq,
        a.begin(), a.end(), b.begin(),
        0,                        // initial value
        std::plus<>(),            // reduction (sum)
        std::multiplies<>()       // transformation (product)
    );

    std::cout << "Dot product: " << dot_product << "\n";
}

int main() 
{
    std_transform();
    std_reduce();
    transform_reduce();
    return 0;
}
