// pg 199
/*
For debugging purposes alone, all three concurrent queues provide limited iterator
support (iterator and const_iterator types). This support is intended solely to
allow us to inspect a queue during debugging. Both iterator and const_iterator
types follow the usual STL conventions for forward iterators. The iteration order is
from least recently pushed to most recently pushed. Modifying a queue invalidates any
iterators that reference it. The iterators are relatively slow. They should be used only for
debugging. An example of usage is shown in Figure 6-10.
*/
/*
    Copyright (c) 2025 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
#include <tbb/concurrent_queue.h>
#include <iostream>

int main() {
  tbb::concurrent_queue<int> queue;
  for( int i=0; i<10; ++i )
    queue.push(i);
  for( tbb::concurrent_queue<int>::const_iterator
       i(queue.unsafe_begin()); 
       i!=queue.unsafe_end();
       ++i )
    std::cout << *i << " ";
  std::cout << std::endl;
  return 0;
}