#pragma once

#include <timers/scoped_timer.h>

// Time from an explicit start() until destruction. Declared before a solver's resources, it
// measures how long releasing them takes once the solver returns.
class DeferredTimer {
public:
  explicit DeferredTimer(const char* name) : name_(name) {}

  ~DeferredTimer() { scoped_timer_stop(&timer_); }

  DeferredTimer(const DeferredTimer&) = delete;
  DeferredTimer& operator=(const DeferredTimer&) = delete;

  void start() { scoped_timer_start(&timer_, name_); }

private:
  const char* name_;
  scoped_timer_t timer_{};
};
