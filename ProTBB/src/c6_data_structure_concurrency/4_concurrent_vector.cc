// Pg 203
#include <iostream>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

void oneway() {
//  Create a vector containing integers
    tbb::concurrent_vector<int> v = {3, 14, 15, 92};

    // Add more integers to vector SERIALLY 
    for( int i = 100; i < 1000; ++i ) {
	v.push_back(i*100+11);
	v.push_back(i*100+22);
	v.push_back(i*100+33);
	v.push_back(i*100+44);
    }

    // Iterate and print values of vector (debug use only)
    for(int n : v) {
      std::cout << n << std::endl;
    }
}

void allways() {
//  Create a vector containing integers
    tbb::concurrent_vector<int> v = {3, 14, 15, 92};

    // Add more integers to vector IN PARALLEL 
    tbb::parallel_for( 100, 999, [&](int i){
	v.push_back(i*100+11);
	v.push_back(i*100+22);
	v.push_back(i*100+33);
	v.push_back(i*100+44);
      });

    // Iterate and print values of vector (debug use only)
    for(int n : v) {
      std::cout << n << std::endl;
    }
}

// While concurrent growing is fundamentally incompatible with
// ideal exception safety, concurrent_vector does offer a
// practical level of exception safety. The element type
// must have a destructor that never throws an exception,
// and if the constructor can throw an exception, then the
// destructor must be nonvirtual and work correctly on
// zero-filled memory.
//
// The push_back(x) method safely appends x to the
// vector. The grow_by(n) method safely appends n
// consecutive elements initialized with T(). Both methods
// return an iterator pointing to the first appended
// element. Each element is initialized with T(). The
// following routine safely appends a C string to a shared
// vector:

void Append( tbb::concurrent_vector<char>& vector,
	     const char* string ) {
  size_t n = strlen(string)+1;
  std :: copy( string, string+n, vector.grow_by(n) );
}

int main() {
  oneway();
  std::cout << std::endl;
  allways();
  return 0;
}
