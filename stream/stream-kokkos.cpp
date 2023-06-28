//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include "Kokkos_Core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

#define STREAM_ARRAY_SIZE 10000
#define STREAM_NTIMES 20

using StreamDeviceArray =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using StreamHostArray = typename StreamDeviceArray::HostMirror;

using StreamIndex = int;
using Policy      = Kokkos::RangePolicy<Kokkos::IndexType<StreamIndex>>;

using DefaultExecSpace = Kokkos::DefaultExecutionSpace;

void perform_set(StreamDeviceArray& a, const double scalar, uint64_t ls, uint64_t ts, uint64_t vs) {
  const int64_t iters_per_team = a.extent(0) / ls;
  const int64_t iters_per_thread= iters_per_team / ts;

    auto policy =
      Kokkos::TeamPolicy<DefaultExecSpace>(ls, ts, vs);
  using team_t = const Kokkos::TeamPolicy<>::member_type;


  Kokkos::parallel_for(
  "set", policy,
  KOKKOS_LAMBDA(team_t &team) {
  const int64_t first_i = team.league_rank() * iters_per_team;
  const int64_t last_i  = first_i + iters_per_team < a.extent(0)
                                  ? first_i + iters_per_team
                                  : a.extent(0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const StreamIndex j){
    const int64_t first_thread_i = team.team_rank() * iters_per_thread;
    const int64_t last_thread_i  = first_thread_i + iters_per_thread < last_i
                                  ? first_thread_i + iters_per_thread
                                  : last_i;
          Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(team,last_thread_i - first_thread_i), [=](const StreamIndex i) {
            a[i] = scalar;
          });
      });
  });

  Kokkos::fence();
}

void perform_copy(StreamDeviceArray& a, StreamDeviceArray& b, uint64_t ls, uint64_t ts, uint64_t vs) {

  const int64_t iters_per_team = a.extent(0) / ls;
  const int64_t iters_per_thread= iters_per_team / ts;

    auto policy =
      Kokkos::TeamPolicy<DefaultExecSpace>(ls, ts, vs);
  using team_t = const Kokkos::TeamPolicy<>::member_type;


  Kokkos::parallel_for(
  "copy", policy,
  KOKKOS_LAMBDA(team_t &team) {
  const int64_t first_i = team.league_rank() * iters_per_team;
  const int64_t last_i  = first_i + iters_per_team < a.extent(0)
                                  ? first_i + iters_per_team
                                  : a.extent(0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const StreamIndex j){
    const int64_t first_thread_i = team.team_rank() * iters_per_thread;
    const int64_t last_thread_i  = first_thread_i + iters_per_thread < last_i
                                  ? first_thread_i + iters_per_thread
                                  : last_i;
          Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(team,last_thread_i - first_thread_i), [=](const StreamIndex i) {
            b[i] = a[i];
          });
      });
  });

  Kokkos::fence();
}

void perform_scale(StreamDeviceArray& b, StreamDeviceArray& c,
                   const double scalar, uint64_t ls, uint64_t ts, uint64_t vs) {
  const int64_t iters_per_team = b.extent(0) / ls;
  const int64_t iters_per_thread= iters_per_team / ts;

  auto policy =
    Kokkos::TeamPolicy<DefaultExecSpace>(ls, ts, vs);
  using team_t = const Kokkos::TeamPolicy<>::member_type;


  Kokkos::parallel_for(
  "scale", policy,
  KOKKOS_LAMBDA(team_t &team) {
  const int64_t first_i = team.league_rank() * iters_per_team;
  const int64_t last_i  = first_i + iters_per_team < b.extent(0)
                                  ? first_i + iters_per_team
                                  : b.extent(0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const StreamIndex j){
    const int64_t first_thread_i = team.team_rank() * iters_per_thread;
    const int64_t last_thread_i  = first_thread_i + iters_per_thread < last_i
                                  ? first_thread_i + iters_per_thread
                                  : last_i;
          Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(team,last_thread_i - first_thread_i), [=](const StreamIndex i) {
            b[i] = scalar * c[i];
          });
      });
  });

  Kokkos::fence();
}

void perform_add(StreamDeviceArray& a, StreamDeviceArray& b,
                 StreamDeviceArray& c, uint64_t ls, uint64_t ts, uint64_t vs) {
  
  const int64_t iters_per_team = a.extent(0) / ls;
  const int64_t iters_per_thread= iters_per_team / ts;

  auto policy =
    Kokkos::TeamPolicy<DefaultExecSpace>(ls, ts, vs);
  using team_t = const Kokkos::TeamPolicy<>::member_type;


  Kokkos::parallel_for(
  "add", policy,
  KOKKOS_LAMBDA(team_t &team) {
  const int64_t first_i = team.league_rank() * iters_per_team;
  const int64_t last_i  = first_i + iters_per_team < a.extent(0)
                                  ? first_i + iters_per_team
                                  : a.extent(0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const StreamIndex j){
    const int64_t first_thread_i = team.team_rank() * iters_per_thread;
    const int64_t last_thread_i  = first_thread_i + iters_per_thread < last_i
                                  ? first_thread_i + iters_per_thread
                                  : last_i;
          Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(team,last_thread_i - first_thread_i), [=](const StreamIndex i) {
            c[i] = a[i] + b[i]; 
          });
      });
  });

  Kokkos::fence();
}

void perform_triad(StreamDeviceArray& a, StreamDeviceArray& b,
                   StreamDeviceArray& c, const double scalar, uint64_t ls, uint64_t ts, uint64_t vs) {
   const int64_t iters_per_team = a.extent(0) / ls;
  const int64_t iters_per_thread= iters_per_team / ts;

  auto policy =
    Kokkos::TeamPolicy<DefaultExecSpace>(ls, ts, vs);
  using team_t = const Kokkos::TeamPolicy<>::member_type;


  Kokkos::parallel_for(
  "triad", policy,
  KOKKOS_LAMBDA(team_t &team) {
  const int64_t first_i = team.league_rank() * iters_per_team;
  const int64_t last_i  = first_i + iters_per_team < a.extent(0)
                                  ? first_i + iters_per_team
                                  : a.extent(0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_i, last_i), [&](const StreamIndex j){
    const int64_t first_thread_i = team.team_rank() * iters_per_thread;
    const int64_t last_thread_i  = first_thread_i + iters_per_thread < last_i
                                  ? first_thread_i + iters_per_thread
                                  : last_i;
          Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(team,last_thread_i - first_thread_i), [=](const StreamIndex i) {
            a[i] = b[i] + scalar * c[i]; 
          });
      });
  });

  Kokkos::fence();
}

int perform_validation(StreamHostArray& a, StreamHostArray& b,
                       StreamHostArray& c, const StreamIndex arraySize,
                       const double scalar) {
  double ai = 1.0;
  double bi = 2.0;
  double ci = 0.0;

  for (StreamIndex i = 0; i < arraySize; ++i) {
    ci = ai;
    bi = scalar * ci;
    ci = ai + bi;
    ai = bi + scalar * ci;
  };

  double aError = 0.0;
  double bError = 0.0;
  double cError = 0.0;

  for (StreamIndex i = 0; i < arraySize; ++i) {
    aError = std::abs(a[i] - ai);
    bError = std::abs(b[i] - bi);
    cError = std::abs(c[i] - ci);
  }

  double aAvgError = aError / (double)arraySize;
  double bAvgError = bError / (double)arraySize;
  double cAvgError = cError / (double)arraySize;

  const double epsilon = 1.0e-13;
  int errorCount       = 0;

  if (std::abs(aAvgError / ai) > epsilon) {
    fprintf(stderr, "Error: validation check on View a failed.\n");
    errorCount++;
  }

  if (std::abs(bAvgError / bi) > epsilon) {
    fprintf(stderr, "Error: validation check on View b failed.\n");
    errorCount++;
  }

  if (std::abs(cAvgError / ci) > epsilon) {
    fprintf(stderr, "Error: validation check on View c failed.\n");
    errorCount++;
  }

  if (errorCount == 0) {
    printf("All solutions checked and verified.\n");
  }

  return errorCount;
}

int run_benchmark(uint64_t size, uint64_t reps, uint64_t ls, uint64_t ts, uint64_t vs) {
  StreamDeviceArray dev_a("a", size);
  StreamDeviceArray dev_b("b", size);
  StreamDeviceArray dev_c("c", size);

  StreamHostArray a = Kokkos::create_mirror_view(dev_a);
  StreamHostArray b = Kokkos::create_mirror_view(dev_b);
  StreamHostArray c = Kokkos::create_mirror_view(dev_c);

  const double scalar = 3.0;

  double setTime   = std::numeric_limits<double>::max();
  double copyTime  = std::numeric_limits<double>::max();
  double scaleTime = std::numeric_limits<double>::max();
  double addTime   = std::numeric_limits<double>::max();
  double triadTime = std::numeric_limits<double>::max();

  Kokkos::parallel_for(
      "init",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             size),
      KOKKOS_LAMBDA(const int i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
      });

  Kokkos::deep_copy(dev_a, a);
  Kokkos::deep_copy(dev_b, b);
  Kokkos::deep_copy(dev_c, c);

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < reps; ++k) {
    timer.reset();
    perform_set(dev_c, 1.5, ls, ts, vs);
    setTime = std::min(setTime, timer.seconds());

    timer.reset();
    perform_copy(dev_a, dev_c, ls, ts, vs);
    copyTime = std::min(copyTime, timer.seconds());

    timer.reset();
    perform_scale(dev_b, dev_c, scalar, ls, ts, vs);
    scaleTime = std::min(scaleTime, timer.seconds());

    timer.reset();
    perform_add(dev_a, dev_b, dev_c, ls, ts, vs); 
    addTime = std::min(addTime, timer.seconds());

    timer.reset();
    perform_triad(dev_a, dev_b, dev_c, scalar, ls, ts, vs);
    triadTime = std::min(triadTime, timer.seconds());
  }

  Kokkos::deep_copy(a, dev_a);
  Kokkos::deep_copy(b, dev_b);
  Kokkos::deep_copy(c, dev_c);

  printf("%lu,%lu,%lu,%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
  ls,ts,vs,
  static_cast<uint64_t>(size),
   1.0e-6 * (double)size * (double)sizeof(double),
   3.0e-6 * (double)size * (double)sizeof(double),
   (1.0e-06 * 1.0 * (double)sizeof(double) * (double)size) / setTime,
   (1.0e-06 * 2.0 * (double)sizeof(double) * (double)size) / copyTime,
   (1.0e-06 * 2.0 * (double)sizeof(double) * (double)size) / scaleTime,
   (1.0e-06 * 3.0 * (double)sizeof(double) * (double)size) / addTime,
   (1.0e-06 * 3.0 * (double)sizeof(double) * (double)size) / triadTime
  );
  return 0;
}

int main(int argc, char* argv[]) {


  uint64_t array_size = STREAM_ARRAY_SIZE;
  uint64_t repetitions = STREAM_NTIMES;
  uint64_t ls = 32, vs = 32, ts = 32;

  array_size = argc > 1 ? atoi(argv[1]) : array_size;
  repetitions = argc > 2 ? atoi(argv[2]): repetitions;
  ls = argc > 3 ? atoi(argv[3]): ls;
  ts = argc > 4 ? atoi(argv[4]): ts;
  vs = argc > 5 ? atoi(argv[5]): vs;

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark(array_size, repetitions, ls, ts, vs);
  Kokkos::finalize();

  return rc;
}
