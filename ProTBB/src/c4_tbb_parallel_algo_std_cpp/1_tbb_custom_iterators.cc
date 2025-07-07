// Pg 122
#include <iostream>
#include <tbb/tbb.h>
#include <ranges>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/transform_iterator.hpp>

//‘counting_iterator’ is not a member of ‘tbb’—is because counting_iterator is not part of the core Intel TBB (oneTBB) 
// library. Instead, it's provided by oneDPL (oneAPI DPC++ Library), which is a separate component that extends TBB
// with C++ standard parallel STL features.
void counting_iterator_demo()
{
   auto range = std::views::iota(0, 10);

    tbb::parallel_for_each(range.begin(), range.end(), [](int i) {
        std::cout << "Index: " << i << "\n";
    });
}

// tbb::make_zip_iterator and related zip iterator utilities were removed from Intel TBB during the transition to oneTBB 2021.
// 📉 What Happened?
// The zip iterator interface was part of older TBB versions (like TBB 2020 and earlier).
// With the release of oneTBB 2021, Intel revamped the library, focusing on a simplified and modernized API.
// As part of this overhaul, experimental or less-used features like tbb::zip_iterator were removed to streamline the codebase and reduce maintenance complexity.
void zipped_iterator_demo()
{
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {10, 20, 30};

    auto begin = boost::make_zip_iterator(boost::make_tuple(a.begin(), b.begin()));
    auto end   = boost::make_zip_iterator(boost::make_tuple(a.end(), b.end()));

    for (auto it = begin; it != end; ++it) {
        int x = boost::get<0>(*it);
        int y = boost::get<1>(*it);
        std::cout << "Sum: " << (x + y) << "\n";
    }
}

// tbb::transform_iterator was indeed part of Intel TBB 2019 as a preview feature, but it was removed in the oneTBB 2021 API revamp. 
// So yes—it’s effectively deprecated and no longer available in modern TBB distributions.
// Alternatives : boost::transform_iterator from the Boost's transform iterator.
//                oneapi::dpl::transform_iterator
void transform_iterator_demo()
{
    std::vector<int> data = {1, 2, 3, 4, 5};
    // Transformation function
    const auto square = [](int x) {
        return x * x;
    };

    // Create transform iterators
    auto begin = boost::make_transform_iterator(data.begin(), square);
    auto end   = boost::make_transform_iterator(data.end(), square);

    // Iterate and print transformed values
    for (auto it = begin; it != end; ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";
}

int main(int argc, char const *argv[])
{
    counting_iterator_demo();
    zipped_iterator_demo();
    transform_iterator_demo();
    return 0;
}
