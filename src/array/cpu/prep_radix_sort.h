#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <dgl/aten/types.h>

#include <stdio.h>

#include <unistd.h>
#include <cstdlib>
#include "radix_sort.h"
#include <dgl/aten/array_ops.h>

using namespace std;
namespace dgl {
namespace aten {

#define COALESCING_PREPROCESSING_VERBOSE 0

class PrepThreadState {
 public:
  PrepThreadState() : numUniqueIndices(0) {}
  uint32_t numUniqueIndices;
  uint32_t dummy[15];
  // uint32_t dummy[13];
};

template <typename Tind, typename Tidx>
void init_scratch(
    std::pair<Tidx, Tidx>* scratch,
    Tind* indices,
    long numIndices) {
#pragma omp parallel for
  for (long i = 0; i < numIndices; i++) {
    scratch[i].first = indices[i];
    scratch[i].second = i;
  }
}

template <typename Tidx>
  IdArray preprocessing(
    long indicesCount,
    long maxIndexValue,
    std::pair<Tidx, Tidx>* scratch) {

  #ifdef TIMER
  unsigned long long t3 = __rdtsc();
  #endif
  // Sorting (parallel) of the src nodes
  auto sortedIndexWithOutputRowPair = radix_sort_pair_parallel(
    scratch, scratch + indicesCount, indicesCount, maxIndexValue);

  #ifdef TIMER
  unsigned long long t4 = __rdtsc();
  #endif
  // cout << "Sort: " << (t4-t3)*1e3/freq<< " ms" << endl;

  // Parallel duplicate removal from here
  int maxThreads = omp_get_max_threads();
  #ifdef TIMER
  unsigned long long t5 = __rdtsc();
  #endif

  PrepThreadState prepThreadState[maxThreads];

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    prepThreadState[tid].numUniqueIndices = 0;

    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        prepThreadState[tid].numUniqueIndices++;
      }
    }
  }

  prepThreadState[0].numUniqueIndices += 1;
  for (int i = 1; i < maxThreads; i++)
    prepThreadState[i].numUniqueIndices +=
        prepThreadState[i - 1].numUniqueIndices;

  long uniqueIndicesCount =
      prepThreadState[maxThreads - 1].numUniqueIndices;

  IdArray t_uniqueIndices = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);
  IdArray t_uniqueIndicesReNum = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);

  long* uniqueIndices = static_cast<long*>(t_uniqueIndices->data);
  long* uniqueIndicesReNum = static_cast<long*>(t_uniqueIndicesReNum->data);
  // addedndum
  uniqueIndices[0] = sortedIndexWithOutputRowPair[0].first;
  uniqueIndicesReNum[0] = 0;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto* tstart =
        (tid == 0 ? uniqueIndices + 1
                  : uniqueIndices +
                 prepThreadState[tid - 1].numUniqueIndices);
    auto* tstartr =
        (tid == 0 ? uniqueIndicesReNum + 1
                  : uniqueIndicesReNum +
                 prepThreadState[tid - 1].numUniqueIndices);

    auto offset = (tid == 0 ? 1
                   : prepThreadState[tid - 1].numUniqueIndices);

    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        *tstartr = offset++;
        *tstart = sortedIndexWithOutputRowPair[i].first;
        tstart++;
        tstartr++;
      }
    }
  }
  return t_uniqueIndices;
}
//////////////////// Preprocessing on Int ////////////////////////////
template <typename Tind, typename Tidx>
void init_scratch2(
  Tidx* scratch,
  Tind* indices,
  long numIndices) {
#pragma omp parallel for
  for (long i = 0; i < numIndices; i++) {
    scratch[i] = indices[i];
  }
}

template <typename Tidx>
  IdArray preprocessing2(
    long indicesCount,
    long maxIndexValue,
    Tidx* scratch) {
  // printf("preprocessing 2.....");fflush(0);
  #ifdef TIMER
  unsigned long long t3 = __rdtsc();
  #endif
  // Sorting (parallel) of the src nodes
  auto sortedIndexWithOutputRowPair = radix_sort_single_parallel(
    scratch, scratch + indicesCount, indicesCount, maxIndexValue);

  #if TIMER
  for (int i = 1; i < indicesCount; i++)
    assert(sortedIndexWithOutputRowPair[i] >= sortedIndexWithOutputRowPair[i-1]);
  printf("checks out\n");fflush(0);
  unsigned long long t4 = __rdtsc();
  #endif
  // cout << "Sort: " << (t4-t3)*1e3/freq<< " ms" << endl;

  // Parallel duplicate removal from here
  int maxThreads = omp_get_max_threads();
  #ifdef TIMER
  unsigned long long t5 = __rdtsc();
  #endif

  PrepThreadState prepThreadState[maxThreads];

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    prepThreadState[tid].numUniqueIndices = 0;

    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i] !=
          sortedIndexWithOutputRowPair[i - 1]) {
        prepThreadState[tid].numUniqueIndices++;
      }
    }
  }

  prepThreadState[0].numUniqueIndices += 1;
  for (int i = 1; i < maxThreads; i++)
    prepThreadState[i].numUniqueIndices +=
        prepThreadState[i - 1].numUniqueIndices;

  long uniqueIndicesCount =
      prepThreadState[maxThreads - 1].numUniqueIndices;

  IdArray t_uniqueIndices = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);
  IdArray t_uniqueIndicesReNum = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);

  long* uniqueIndices = static_cast<long*>(t_uniqueIndices->data);
  long* uniqueIndicesReNum = static_cast<long*>(t_uniqueIndicesReNum->data);
  // addedndum
  uniqueIndices[0] = sortedIndexWithOutputRowPair[0];
  uniqueIndicesReNum[0] = 0;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto* tstart =
        (tid == 0 ? uniqueIndices + 1
                  : uniqueIndices +
                 prepThreadState[tid - 1].numUniqueIndices);
    auto* tstartr =
        (tid == 0 ? uniqueIndicesReNum + 1
                  : uniqueIndicesReNum +
                 prepThreadState[tid - 1].numUniqueIndices);

    auto offset = (tid == 0 ? 1
                   : prepThreadState[tid - 1].numUniqueIndices);

    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i] !=
          sortedIndexWithOutputRowPair[i - 1]) {
        *tstartr = offset++;
        *tstart = sortedIndexWithOutputRowPair[i];
        tstart++;
        tstartr++;
      }
    }
  }
  return t_uniqueIndices;
}


///////////////////// preprocessing extended //////////////////////////
template <typename Tidx>
std::pair<Tidx, Tidx>* sortPair(
    long indicesCount,
    long maxIndexValue,
    std::pair<Tidx, Tidx>* scratch) {

  #ifdef TIMER
  unsigned long long t3 = __rdtsc();
  #endif
  auto sortedIndexWithOutputRowPair = radix_sort_pair_parallel(
    scratch, scratch + indicesCount, indicesCount, maxIndexValue);

  #ifdef TIMER
  unsigned long long t4 = __rdtsc();
  #endif

  return sortedIndexWithOutputRowPair;
}

template <typename Tidx>
std::tuple<IdArray, IdArray> preprocessing_ext(
    long indicesCount,
    long maxIndexValue,
    std::pair<Tidx, Tidx>* scratch) {

  #ifdef TIMER
  unsigned long long t3 = __rdtsc();
  #endif
  auto sortedIndexWithOutputRowPair = radix_sort_pair_parallel(
    scratch, scratch + indicesCount, indicesCount, maxIndexValue);

  #ifdef TIMER
  unsigned long long t4 = __rdtsc();
  #endif

  int maxThreads = omp_get_max_threads();
  #ifdef TIMER
  unsigned long long t5 = __rdtsc();
  #endif

  PrepThreadState prepThreadState[maxThreads];

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    prepThreadState[tid].numUniqueIndices = 0;
    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        prepThreadState[tid].numUniqueIndices++;
      }
    }
  }

  prepThreadState[0].numUniqueIndices += 1;
  for (int i = 1; i < maxThreads; i++)
    prepThreadState[i].numUniqueIndices +=
        prepThreadState[i - 1].numUniqueIndices;

  long uniqueIndicesCount =
      prepThreadState[maxThreads - 1].numUniqueIndices;

  IdArray t_uniqueIndices = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);

  IdArray t_uniqueIndicesReNum = NewIdArray(uniqueIndicesCount, DGLContext{kDGLCPU, 0}, sizeof(long)*8);

  long* uniqueIndices = static_cast<long*>(t_uniqueIndices->data);
  long* uniqueIndicesReNum = static_cast<long*>(t_uniqueIndicesReNum->data);
  // addedndum
  uniqueIndices[0] = sortedIndexWithOutputRowPair[0].first;
  uniqueIndicesReNum[0] = sortedIndexWithOutputRowPair[0].second;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto* tstart =
        (tid == 0 ? uniqueIndices + 1
                  : uniqueIndices +
                 prepThreadState[tid - 1].numUniqueIndices);
    auto* tstartind =
        (tid == 0 ? uniqueIndicesReNum + 1
                  : uniqueIndicesReNum +
                 prepThreadState[tid - 1].numUniqueIndices);

    auto offset = (tid == 0 ? 1
                   : prepThreadState[tid - 1].numUniqueIndices);

    int index = -1;
    #pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      index = i;
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        *tstartind = sortedIndexWithOutputRowPair[i].second;
        *tstart = sortedIndexWithOutputRowPair[i].first;
        tstart++;
        tstartind++;
      }
    } // for
    // #pragma omp barrier
    auto a = tstart - 1;
    auto b = tstartind - 1;
    while((index + 1) < indicesCount) {
      if (*a == sortedIndexWithOutputRowPair[index + 1].first) {
        if(*b > sortedIndexWithOutputRowPair[index + 1].second)
          *b = sortedIndexWithOutputRowPair[index + 1].second;
      }
      else
        break;
      index ++;
    }
  }

  return std::make_tuple(t_uniqueIndices, t_uniqueIndicesReNum);
}
}
}