/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/array_utils.h
 * @brief Utility classes and functions for DGL arrays.
 */
#ifndef DGL_ARRAY_CPU_ARRAY_UTILS_H_
#define DGL_ARRAY_CPU_ARRAY_UTILS_H_

#include "prep_radix_sort.h"
#include <x86intrin.h>
#include <dgl/aten/types.h>
#include <parallel_hashmap/phmap.h>

#include <unordered_map>
#include <utility>
#include <vector>
#include <valarray>

#include "../../c_api_common.h"

namespace dgl {
namespace aten {

// new dl
template <typename IdType>
class IdHashMapSync {
  class PrepThreadState {
 public:
  PrepThreadState() : uniqueIndices(0) {}
  // uint32_t numUniqueIndices;
  uint32_t uniqueIndices;
  uint32_t dummy[15];
};
 public:
  // default ctor
  IdHashMapSync(): filter_(kFilterSize, false) {
    maxThreads = omp_get_max_threads();
    size = offset = 0;
  }

  explicit IdHashMapSync(const std::vector<IdArray>& hashmap, std::vector<IdArray>& new_nodes): filter_(kFilterSize, false) {
    for (size_t i = 0; i < hashmap.size(); i++) {
      hash_table.push_back(IdArray(hashmap[i]));
      if(new_nodes[i].defined())
        values.push_back(IdArray(new_nodes[i]));
      else
        values.push_back(NewIdArray(0, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8));
      size.push_back(values[i]->shape[0]);
      offset.push_back(values[i]->shape[0]);
    }
    maxThreads = omp_get_max_threads();
  }

  std::pair<std::vector<IdArray>, std::vector<IdArray>> return_mapping_and_nodes() {
    return std::make_pair(hash_table, values);
  }

  // Return true if the given id is contained in this hashmap.
  // Update the hashmap with given id array.
  // The id array could contain duplicates.

  void UpdateSrc(IdArray ids, int map_id, IdType max_val, IdArray scratch_mem) {
    #if TIMER
    uint64_t tic = _rdtsc();
    printf("Update starts...\n");
    #endif

    IdType *hash_table_ptr = static_cast<IdType*>(hash_table[map_id]->data);
    IdType* ids_data = static_cast<IdType*>(ids->data);
    int64_t len = ids->shape[0];

    //std::valarray<IdType> ids_val(ids_data, len);
    //IdType maxInd = ids_val.max();
    IdType maxInd = max_val;
    #if 0
    IdArray t_scratch = NewIdArray(4*len, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
    // std::tuple<IdArray, IdArray> ret;
    std::pair<IdType, IdType>* scratch = static_cast<std::pair<IdType, IdType>*>(t_scratch->data);
    init_scratch<IdType, IdType>(scratch, ids_data, len);
    IdArray t_uniqueIndices = preprocessing<IdType>(len, maxInd, scratch);
    #else
    //IdArray t_scratch = NewIdArray(2*len, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
    IdType* scratch = static_cast<IdType*>(scratch_mem->data);
    //init_scratch2<IdType, IdType>(scratch, ids_data, len);
    IdArray t_uniqueIndices = preprocessing2<IdType>(len, maxInd, scratch);
    #endif

    PrepThreadState count[maxThreads];
    PrepThreadState countAcc[maxThreads];
    IdType *uniq_id = static_cast<IdType*>(t_uniqueIndices->data);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (int i=0; i<t_uniqueIndices->shape[0]; i++) {
        auto id = uniq_id[i];
        assert(id < ht_size && id >= 0);
        if (hash_table_ptr[id] == -1) {
          count[tid].uniqueIndices++;
        }
        else {
          uniq_id[i] = -1;
        }
      }
    }

    countAcc[0].uniqueIndices = offset[map_id];
    for(int i=1; i<maxThreads; i++) {
      countAcc[i].uniqueIndices = count[i-1].uniqueIndices + countAcc[i-1].uniqueIndices;
    }
    size[map_id] = count[maxThreads-1].uniqueIndices + countAcc[maxThreads-1].uniqueIndices;
    IdArray valuesTmp = NewIdArray(size[map_id], DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
    IdType* values_data_tmp = static_cast<IdType*>(valuesTmp->data);
    IdType* values_data = static_cast<IdType*>(values[map_id]->data);
    #pragma omp parallel for schedule(static)
    for (int l=0; l<offset[map_id]; l++) values_data_tmp[l] = values_data[l];

    values[map_id] = valuesTmp;
    values_data = static_cast<IdType*>(values[map_id]->data);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (int i=0; i<t_uniqueIndices->shape[0]; i++) {
        auto id = uniq_id[i];
        if (id != -1) { // && hash_table[id] == -1) count[tid].uniqueIndices++;
          hash_table_ptr[id] = countAcc[tid].uniqueIndices;
          values_data[countAcc[tid].uniqueIndices] = id;
          countAcc[tid].uniqueIndices++;
        }
      }
    }
    offset[map_id] = size[map_id];

    #if TIMER
    printf("Update done...sizeof IdType: %d\n", sizeof(IdType));
    uint64_t toc = _rdtsc();
    printf("UpdateSrc time: %ld\n", toc - tic);
    #endif
  }

  IdType MapHash(IdType id, IdType default_val, int map_id) const {
    IdType *hash_table_ptr = static_cast<IdType*>(hash_table[map_id]->data);
    auto ret = hash_table_ptr[id];
    if (ret != -1)
      return ret;
    else
      return default_val;
  }

  // Return the new id of each id in the given array.
  IdArray MapHash(IdArray ids, IdType default_val, int map_id) const {
    #if TIMER
    uint64_t tic = _rdtsc();
    #endif

    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdArray values = NewIdArray(len, ids->ctx, ids->dtype.bits);
    IdType* values_data = static_cast<IdType*>(values->data);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < len; ++i)
      values_data[i] = MapHash(ids_data[i], default_val, map_id);

    #if TIMER
    uint64_t toc = _rdtsc();
    printf("MapHash time: %ld\n", toc - tic);
    #endif
    return values;
  }

  // placeholders, as other array cpu routines throw errors because of absence of these names
  void Reserve(const int64_t size) {
      // oldv2newv_.reserve(size);
  }

  // Update the hashmap with given id array.
  // The id array could contain duplicates.
  void Update(IdArray ids) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      // phmap::flat_hash_map::insert assures that an insertion will not happen if the
      // key already exists.
      oldv2newv_.insert({id, oldv2newv_.size()});
      filter_[id & kFilterMask] = true;
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

  // Return the new id of each id in the given array.
  IdArray Map(IdArray ids, IdType default_val) const {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdArray values = NewIdArray(len, ids->ctx, ids->dtype.bits);
    IdType* values_data = static_cast<IdType*>(values->data);
    for (int64_t i = 0; i < len; ++i)
      values_data[i] = Map(ids_data[i], default_val);
    return values;
  }

  // Return all the old ids collected so far, ordered by new id.
  IdArray Values() const {
    IdArray values = NewIdArray(oldv2newv_.size(), DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
    return values;
  }

  inline size_t Size() const {
    return oldv2newv_.size();
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  phmap::parallel_flat_hash_map<IdType, IdType> oldv2newv_;
  std::vector<IdArray> hash_table, values;
  std::vector<long> offset, size;
  int32_t maxThreads;
  static constexpr long ht_size = 112000000;  // make this as param
};

/**
 * @brief A hashmap that maps each ids in the given array to new ids starting
 * from zero.
 *
 * Useful for relabeling integers and finding unique integers.
 *
 * Usually faster than std::unordered_map in existence checking.
 */
template <typename IdType>
class IdHashMap {
 public:
  // default ctor
  IdHashMap() : filter_(kFilterSize, false) {}

  // Construct the hashmap using the given id array.
  // The id array could contain duplicates.
  // If the id array has no duplicates, the array will be relabeled to
  // consecutive integers starting from 0.
  explicit IdHashMap(IdArray ids) : filter_(kFilterSize, false) {
    oldv2newv_.reserve(ids->shape[0]);
    Update(ids);
  }

  // copy ctor
  IdHashMap(const IdHashMap& other) = default;

  void Reserve(const int64_t size) { oldv2newv_.reserve(size); }

  // Update the hashmap with given id array.
  // The id array could contain duplicates.
  void Update(IdArray ids) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      // phmap::flat_hash_map::insert assures that an insertion will not happen
      // if the key already exists.
      oldv2newv_.insert({id, oldv2newv_.size()});
      filter_[id & kFilterMask] = true;
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

  // Return the new id of each id in the given array.
  IdArray Map(IdArray ids, IdType default_val) const {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdArray values = NewIdArray(len, ids->ctx, ids->dtype.bits);
    IdType* values_data = static_cast<IdType*>(values->data);
    for (int64_t i = 0; i < len; ++i)
      values_data[i] = Map(ids_data[i], default_val);
    return values;
  }

  // Return all the old ids collected so far, ordered by new id.
  IdArray Values() const {
    IdArray values = NewIdArray(
        oldv2newv_.size(), DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
    IdType* values_data = static_cast<IdType*>(values->data);
    for (auto pair : oldv2newv_) values_data[pair.second] = pair.first;
    return values;
  }

  inline size_t Size() const { return oldv2newv_.size(); }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up
  // lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  phmap::flat_hash_map<IdType, IdType> oldv2newv_;
};

/**
 * @brief Hash type for building maps/sets with pairs as keys.
 */
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ARRAY_UTILS_H_
