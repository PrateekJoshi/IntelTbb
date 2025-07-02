## Build 

```
mkdir build ; cd build
cmake ..
make -j
```

## Test

```
# CTEST_OUTPUT_ON_FAILURE=1 will show gtest output if a test fails
# OR
# ctest -V
CTEST_OUTPUT_ON_FAILURE=1 make test
```

## Coverage

```
make coverage
```

## Generate docs

```
# sudo apt-get update
# sudo apt install doxygen
# sudo apt-get install graphviz   ( for dot files showing dependency tree in docs )
make GenerateDocs -j
```

## Static Code Analysis

```
# https://www.youtube.com/watch?v=8RSxQ8sluG0
# sudo apt install clang-tidy
# Run below from project root directory
# Check and exceptions can be configured at : .clang-tidy
run-clang-tidy -use-color -p build ./src/*
```

## Code formatting and check

```
# Check if code is formatted correctly
make check_cpp_format

# Format all cpp files and header of project 
make format_cpp
```

## Reference


## Troubleshooting
