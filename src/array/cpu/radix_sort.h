#ifndef _PCL_RADIX_SORT_
#define _PCL_RADIX_SORT_

// #include "sort_headers.h"
#include <omp.h>
#include <limits>
#include <utility>
#ifdef DEBUG
#include <sys/time.h>
#endif

#ifndef BKT_BITS
#define BKT_BITS 8
#endif

template <typename Tind>
using Key_Value_Pair = std::pair<Tind, Tind>;

template <typename Tind>
Key_Value_Pair<Tind>* radix_sort_pair_parallel(
    Key_Value_Pair<Tind>* inp_buf,
    Key_Value_Pair<Tind>* tmp_buf,
    int64_t elements_count,
    int64_t max_value) {
  constexpr int bkt_bits = BKT_BITS;
  constexpr int nbkts = (1 << bkt_bits);
  constexpr int bkt_mask = (nbkts - 1);

  int maxthreads = omp_get_max_threads();
  maxthreads = ((maxthreads + 15) / 16) * 16;
  int histogram[nbkts * maxthreads], histogram_t[nbkts * maxthreads], histogram_ps[nbkts * maxthreads];
  int histogram_gbl[nbkts][16];
  if (max_value == 0)
    return inp_buf;
  int num_bits = 64;
  if (sizeof(Tind) == 8 && max_value > std::numeric_limits<int>::max()) {
    num_bits = sizeof(Tind) * 8 - __builtin_clzll(max_value);
  } else {
    num_bits = 32 - __builtin_clz((unsigned int)max_value);
  }

  int num_passes = (num_bits + bkt_bits - 1) / bkt_bits;

#ifdef DEBUG
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);
#endif

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[nbkts * tid];
    int* local_histogram_ps = &histogram_ps[nbkts * tid];
    int elements_count_4 = elements_count / 4 * 4;
    Key_Value_Pair<Tind>* input = inp_buf;
    Key_Value_Pair<Tind>* output = tmp_buf;

    for (unsigned int pass = 0; pass < (unsigned int)num_passes; pass++) {
#ifdef DEBUG
      struct timeval tv1, tv2, tv3, tv4;
      gettimeofday(&tv1, NULL);
#endif
      // Step 1: compute histogram
      // Reset histogram
      for (int i = 0; i < nbkts; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i].first;
        Tind val_2 = input[i + 1].first;
        Tind val_3 = input[i + 2].first;
        Tind val_4 = input[i + 3].first;

        local_histogram[(val_1 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_2 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_3 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_4 >> (pass * bkt_bits)) & bkt_mask]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i].first;
          local_histogram[(val >> (pass * bkt_bits)) & bkt_mask]++;
        }
      }
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv2, NULL);
#endif
      // Step 2: parallel prefix sum
#pragma omp for
      for (int bins = 0; bins < nbkts; bins++) {
        int sum = 0;
        for (int t = 0; t < nthreads; t++) {
          histogram_t[bins * maxthreads + t] = sum;
          sum += histogram[t * nbkts + bins];
        }
        histogram_gbl[bins][0] = sum;
      }
      int sum = 0;
      local_histogram_ps[0] = histogram_t[tid];
      for (int bins = 1; bins < nbkts; bins++) {
        sum += histogram_gbl[bins-1][0];
        local_histogram_ps[bins] = histogram_t[bins * maxthreads + tid] + sum;
      }
      sum += histogram_gbl[nbkts-1][0];
      if (sum != elements_count) {
        printf("Error1!\n");
        exit(123);
      }
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv3, NULL);
#endif

      // Step 3: scatter
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i].first;
        Tind val_2 = input[i + 1].first;
        Tind val_3 = input[i + 2].first;
        Tind val_4 = input[i + 3].first;
        Tind bin_1 = (val_1 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_2 = (val_2 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_3 = (val_3 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_4 = (val_4 >> (pass * bkt_bits)) & bkt_mask;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i].first;
          int pos = local_histogram_ps[(val >> (pass * bkt_bits)) & bkt_mask]++;
          output[pos] = input[i];
        }
      }

      Key_Value_Pair<Tind>* temp = input;
      input = output;
      output = temp;
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv4, NULL);
      if (tid == 0) {
	double t1 = (tv4.tv_usec*1e-6 + tv4.tv_sec) - (tv1.tv_usec*1e-6 + tv1.tv_sec);
	double t2 = (tv2.tv_usec*1e-6 + tv2.tv_sec) - (tv1.tv_usec*1e-6 + tv1.tv_sec);
	double t3 = (tv3.tv_usec*1e-6 + tv3.tv_sec) - (tv2.tv_usec*1e-6 + tv2.tv_sec);
	double t4 = (tv4.tv_usec*1e-6 + tv4.tv_sec) - (tv3.tv_usec*1e-6 + tv3.tv_sec);


	printf("pass = %d, elements = %ld   time = %.6f  %.6f  %.6f %.6f\n",pass,elements_count,t1,t2,t3,t4);
      }
#endif
    }
  }
#ifdef DEBUG
  gettimeofday(&tve, NULL);
  double tv = (tve.tv_usec*1e-6 + tve.tv_sec) - (tvs.tv_usec*1e-6 + tvs.tv_sec);
  printf("trs = %.6f\n",tv);
#endif
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

template <typename Tind>
Tind* radix_sort_single_parallel(
  Tind* inp_buf,
  Tind* tmp_buf,
  int64_t elements_count,
  int64_t max_value) {
  constexpr int bkt_bits = BKT_BITS;
  constexpr int nbkts = (1 << bkt_bits);
  constexpr int bkt_mask = (nbkts - 1);

  int maxthreads = omp_get_max_threads();
  maxthreads = ((maxthreads + 15) / 16) * 16;
  int histogram[nbkts * maxthreads], histogram_t[nbkts * maxthreads], histogram_ps[nbkts * maxthreads];
  int histogram_gbl[nbkts][16];
  if (max_value == 0)
    return inp_buf;
  int num_bits = 64;
  if (sizeof(Tind) == 8 && max_value > std::numeric_limits<int>::max()) {
    num_bits = sizeof(Tind) * 8 - __builtin_clzll(max_value);
  } else {
    num_bits = 32 - __builtin_clz((unsigned int)max_value);
  }

  int num_passes = (num_bits + bkt_bits - 1) / bkt_bits;

#ifdef DEBUG
  struct timeval tvs, tve;
  gettimeofday(&tvs, NULL);
#endif

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[nbkts * tid];
    int* local_histogram_ps = &histogram_ps[nbkts * tid];
    int elements_count_4 = elements_count / 4 * 4;
    Tind* input = inp_buf;
    Tind* output = tmp_buf;

    for (unsigned int pass = 0; pass < (unsigned int)num_passes; pass++) {
#ifdef DEBUG
      struct timeval tv1, tv2, tv3, tv4;
      gettimeofday(&tv1, NULL);
#endif
      // Step 1: compute histogram
      // Reset histogram
      for (int i = 0; i < nbkts; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i];
        Tind val_2 = input[i + 1];
        Tind val_3 = input[i + 2];
        Tind val_4 = input[i + 3];

        local_histogram[(val_1 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_2 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_3 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_4 >> (pass * bkt_bits)) & bkt_mask]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i];
          local_histogram[(val >> (pass * bkt_bits)) & bkt_mask]++;
        }
      }
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv2, NULL);
#endif
      // Step 2: parallel prefix sum
#pragma omp for
      for (int bins = 0; bins < nbkts; bins++) {
        int sum = 0;
        for (int t = 0; t < nthreads; t++) {
          histogram_t[bins * maxthreads + t] = sum;
          sum += histogram[t * nbkts + bins];
        }
        histogram_gbl[bins][0] = sum;
      }
      int sum = 0;
      local_histogram_ps[0] = histogram_t[tid];
      for (int bins = 1; bins < nbkts; bins++) {
        sum += histogram_gbl[bins-1][0];
        local_histogram_ps[bins] = histogram_t[bins * maxthreads + tid] + sum;
      }
      sum += histogram_gbl[nbkts-1][0];
      if (sum != elements_count) {
        printf("Error1!\n");
        exit(123);
      }
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv3, NULL);
#endif

      // Step 3: scatter
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i];
        Tind val_2 = input[i + 1];
        Tind val_3 = input[i + 2];
        Tind val_4 = input[i + 3];
        Tind bin_1 = (val_1 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_2 = (val_2 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_3 = (val_3 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_4 = (val_4 >> (pass * bkt_bits)) & bkt_mask;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i];
          int pos = local_histogram_ps[(val >> (pass * bkt_bits)) & bkt_mask]++;
          output[pos] = input[i];
        }
      }

      Tind* temp = input;
      input = output;
      output = temp;
#pragma omp barrier
#ifdef DEBUG
      gettimeofday(&tv4, NULL);
      if (tid == 0) {
	double t1 = (tv4.tv_usec*1e-6 + tv4.tv_sec) - (tv1.tv_usec*1e-6 + tv1.tv_sec);
	double t2 = (tv2.tv_usec*1e-6 + tv2.tv_sec) - (tv1.tv_usec*1e-6 + tv1.tv_sec);
	double t3 = (tv3.tv_usec*1e-6 + tv3.tv_sec) - (tv2.tv_usec*1e-6 + tv2.tv_sec);
	double t4 = (tv4.tv_usec*1e-6 + tv4.tv_sec) - (tv3.tv_usec*1e-6 + tv3.tv_sec);


	printf("pass = %d, elements = %ld   time = %.6f  %.6f  %.6f %.6f\n",pass,elements_count,t1,t2,t3,t4);
      }
#endif
    }
  }
#ifdef DEBUG
  gettimeofday(&tve, NULL);
  double tv = (tve.tv_usec*1e-6 + tve.tv_sec) - (tvs.tv_usec*1e-6 + tvs.tv_sec);
  printf("trs = %.6f\n",tv);
#endif
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

#endif