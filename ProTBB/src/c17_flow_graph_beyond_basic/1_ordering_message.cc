// Pg 483
#include <tbb/tbb.h>

#include <iostream>

struct Message
{
  size_t my_seq_no;
  std::string my_string;
  Message(int i) : my_seq_no(i), my_string(std::to_string(i)) {}
};

using MessagePtr = std::shared_ptr<Message>;

void outOfOrder()
{
  const int N = 10;
  tbb::flow::graph g;
  tbb::flow::function_node first_node{g, tbb::flow::unlimited, [](const MessagePtr& m)
                                      {
                                        m->my_string += " no sequencer";
                                        return m;
                                      }};
  tbb::flow::function_node last_node{g, tbb::flow::serial, [](const MessagePtr& m)
                                     {
                                       std::cout << m->my_string << std::endl;
                                       return 0;
                                     }};
  tbb::flow::make_edge(first_node, last_node);

  for (int i = 0; i < N; ++i) first_node.try_put(std::make_shared<Message>(i));
  g.wait_for_all();
}


/*
This flow graph demonstrates how to preserve message ordering using Intel TBB’s sequencer_node. Let’s walk through it step by step and explain the purpose of each node and connection.
🧠 Overview
This graph processes a sequence of Message objects, each with a unique my_seq_no. The goal is to:
Modify each message.
Ensure messages are delivered to the final node in order.
Print them one by one.

Type: sequencer_node<MessagePtr>
Purpose: Buffers messages and releases them in ascending order of my_seq_no.
Key Feature: Ensures ordered delivery even if upstream nodes are parallel.

Type: function_node<MessagePtr, int>
Concurrency: serial — only one message at a time.
Policy: rejecting — rejects new messages if busy.
Purpose: Prints the message string and returns a dummy int.

⚠️ Note: rejecting is deprecated in newer oneTBB versions. You can replace it with a serial node and a limiter_node if needed.
*/
void orderWithSequencer()
{
  const int N = 10;
  tbb::flow::graph g;
  tbb::flow::function_node first_node{g, tbb::flow::unlimited, [](const MessagePtr& m)
                                      {
                                        m->my_string += " with sequencer";
                                        return m;
                                      }};
  tbb::flow::sequencer_node sequencer(g, [](const MessagePtr& m) { return m->my_seq_no; });
  tbb::flow::function_node<MessagePtr, int, tbb::flow::rejecting> last_node{g, tbb::flow::serial, [](MessagePtr m)
                                                                            {
                                                                              std::cout << m->my_string << std::endl;
                                                                              return 0;
                                                                            }};
  tbb::flow::make_edge(first_node, sequencer);
  tbb::flow::make_edge(sequencer, last_node);

  for (int i = 0; i < N; ++i) first_node.try_put(std::make_shared<Message>(i));
  g.wait_for_all();
}

/*
sequencer: Multifunction Node for Ordering
Concurrency: serial — ensures deterministic ordering.

Logic:
Stores each incoming message in vector v at index my_seq_no.
Checks if the next expected message (seq_i) is ready.
If yes, emits it via try_put() and increments seq_i.
*/
void orderWithMulti()
{
  const int N = 10;
  tbb::flow::graph g;
  tbb::flow::function_node first_node{g, tbb::flow::unlimited, [](const MessagePtr& m)
                                      {
                                        m->my_string += " with multifunction_node";
                                        return m;
                                      }};

  using MFNSequencer = tbb::flow::multifunction_node<MessagePtr, std::tuple<MessagePtr>>;
  using MFNPorts = typename MFNSequencer::output_ports_type;

  int seq_i = 0;
  std::vector<MessagePtr> v{(const unsigned)N, MessagePtr{}};

  MFNSequencer sequencer{g, tbb::flow::serial, [&seq_i, &v](MessagePtr m, MFNPorts& p)
                         {
                           v[m->my_seq_no] = m;
                           while (seq_i < N && v[seq_i].use_count())
                           {
                             std::get<0>(p).try_put(v[seq_i++]);
                           }
                         }};

  tbb::flow::function_node last_node{g, tbb::flow::serial, [](const MessagePtr& m)
                                     {
                                       std::cout << m->my_string << std::endl;
                                       return 0;
                                     }};
  tbb::flow::make_edge(first_node, sequencer);
  tbb::flow::make_edge(sequencer, last_node);

  for (int i = 0; i < N; ++i) first_node.try_put(std::make_shared<Message>(i));
  g.wait_for_all();
}

static void warmupTBB();

int main(int argc, char* argv[])
{
  warmupTBB();
  tbb::tick_count t0 = tbb::tick_count::now();
  outOfOrder();
  auto t_ooo = (tbb::tick_count::now() - t0).seconds();

  t0 = tbb::tick_count::now();
  orderWithSequencer();
  auto t_seq = (tbb::tick_count::now() - t0).seconds();

  t0 = tbb::tick_count::now();
  orderWithMulti();
  auto t_multi = (tbb::tick_count::now() - t0).seconds();

  std::cout << "OOO time == " << t_ooo << "\n";
  std::cout << "sequencer time == " << t_seq << "\n";
  std::cout << "multifunction_node time == " << t_multi << "\n";
  return 0;
}

static void warmupTBB()
{
  // This is a simple loop that should get workers started.
  // oneTBB creates workers lazily on first use of the library
  // so this hides the startup time when looking at trivial
  // examples that do little real work.
  tbb::parallel_for(0, tbb::info::default_concurrency(),
                    [=](int)
                    {
                      tbb::tick_count t0 = tbb::tick_count::now();
                      while ((tbb::tick_count::now() - t0).seconds() < 0.01);
                    });
}
