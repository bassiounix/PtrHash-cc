#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>
#include <chrono>

class Timer {
public:
  Timer() : startPoint(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {
    auto endPoint = std::chrono::high_resolution_clock::now();
    auto start =
        std::chrono::time_point_cast<std::chrono::nanoseconds>(startPoint)
            .time_since_epoch()
            .count();
    auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endPoint)
                   .time_since_epoch()
                   .count();

    auto duration = end - start;
    std::cout << "TimerLog: " << duration << " ns" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> startPoint;
};

#endif  // TIMER_H_
