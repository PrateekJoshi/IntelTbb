// Pg 327

#include <tbb/tbb.h>

const int default_P = tbb::info::default_concurrency();
void doWork(double seconds);

void arenaGlobalControlExplicitArena(int p) {
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, p);

  tbb::task_arena a{2*tbb::info::default_concurrency()};

  a.execute([]() {
    tbb::parallel_for(0, 
                      10*tbb::info::default_concurrency(), 
                      [](int) { doWork(0.01); });
  });
}

#include <atomic>
#include <iostream>
#include <vector>

std::atomic<int> next_tid;
tbb::enumerable_thread_specific<int> my_tid(-1);
std::vector<std::atomic<int>> tid_participation(2*default_P);

void noteParticipation() {
  auto& t = my_tid.local();
  if (t == -1) {
    t = next_tid++;
  }
  ++tid_participation[t];
}

void doWork(double seconds) {
  noteParticipation();
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < seconds);
}

void clearParticipation() {
  next_tid = 0;
  my_tid.clear();
  for (auto& p : tid_participation)
    p = 0;
}

void dumpParticipation(int p) {
  int sum = tid_participation[0];
  std::cout << "[" << tid_participation[0];
  for (int i = 1; i < p; ++i) {
    sum += tid_participation[i];
    std::cout << ", " << tid_participation[i];
  }
  for (int i = p; i < 2*default_P; ++i) 
    std::cout << ", -";
  std::cout << "]\n" 
            << "sum == " << sum  << "\n"
            << "expected sum " << 10*default_P << "\n\n";
  clearParticipation();
}

int main() {
  arenaGlobalControlExplicitArena(default_P);
  dumpParticipation(default_P);
  arenaGlobalControlExplicitArena(default_P/2);
  dumpParticipation(default_P/2);
  arenaGlobalControlExplicitArena(2*default_P);
  dumpParticipation(2*default_P);
  return 0;
}