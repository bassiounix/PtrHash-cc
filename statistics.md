# Statistics

## `std::unordered_map`

- Total Buckets: 2357
- Used Buckets: 1124
- Total Items: 1458
- Used Slots: 0.476877
- Load Factor: 0.618583
- Max Bucket Size: 4
- Total Memory Used (bytes): 61040 = 60KB

## PtrHash

- Total Buckets: 1458
- Used Buckets: 1458
- Total Items: 1458
- Used Slots: 1
- Load Factor: 1
- Max Bucket Size: 1
- Total Memory Used: 11664 bytes + 489 pilots (bytes) = 12KB

## Custom Table: 1458 Max Items

### Using `std::hash<K>`

- Total Buckets: 1458
- Used Buckets: 961
- Total Items: 1458
- Used Slots: 0.659122
- Load Factor: 1
- Max Bucket Size: 4
- Total Memory Used (bytes): 47472 = 47KB

### Using `hash21`

- Total Buckets: 1458
- Used Buckets: 905
- Total Items: 1458
- Used Slots: 0.620713
- Load Factor: 1
- Max Bucket Size: 6
- Total Memory Used (bytes): 47600 = 47KB

### Using `hash21_fast`

- Total Buckets: 1458
- Used Buckets: 939
- Total Items: 1458
- Used Slots: 0.644033
- Load Factor: 1
- Max Bucket Size: 6
- Total Memory Used (bytes): 47424 = 47KB

## Custom Table: 2000 Max Items

### Using `std::hash<K>`

- Total Buckets: 2000
- Used Buckets: 1101
- Total Items: 1458
- Used Slots: 0.5505
- Load Factor: 0.729
- Max Bucket Size: 4
- Total Memory Used (bytes): 59968 = 59KB

### Using `hash21`

- Total Buckets: 2000
- Used Buckets: 1032
- Total Items: 1458
- Used Slots: 0.516
- Load Factor: 0.729
- Max Bucket Size: 5
- Total Memory Used (bytes): 60240 = 59KB

### Using `hash21_fast`

- Total Buckets: 2000
- Used Buckets: 1015
- Total Items: 1458
- Used Slots: 0.5075
- Load Factor: 0.729
- Max Bucket Size: 6
- Total Memory Used (bytes): 60232 = 59KB

## Used Implementations

```cpp
static inline uint32_t hash21(wint_t key) {
  // Mix the 21-bit key
  key ^= key >> 16;
  key *= 0x7feb352d;
  key ^= key >> 15;
  key *= 0x846ca68b;
  key ^= key >> 16;

  return key % N;
}

static inline uint32_t hash21_fast(wint_t key) {
  key *= 2654435761u; // Knuth multiplicative constant
  return (key ^ (key >> 16)) % N;
}

template <typename K, typename V>
void printUnorderedMapStats(const std::unordered_map<K, V>& m) {
    size_t N = m.bucket_count();
    size_t usedBuckets = 0;
    size_t totalItems = 0;
    size_t maxBucketSize = 0;

    for (size_t i = 0; i < N; ++i) {
        size_t bucketSize = m.bucket_size(i);
        if (bucketSize > 0) {
            usedBuckets++;
            totalItems += bucketSize;
            if (bucketSize > maxBucketSize) {
                maxBucketSize = bucketSize;
            }
        }
    }

    double loadFactor = static_cast<double>(totalItems) / N;
    double usedSlots = static_cast<double>(usedBuckets) / N;

    std::cout << "Total Buckets: " << N << '\n';
    std::cout << "Used Buckets: " << usedBuckets << '\n';
    std::cout << "Total Items: " << totalItems << '\n';
    std::cout << "Used Slots: " << usedSlots << '\n';
    std::cout << "Load Factor: " << loadFactor << '\n';
    std::cout << "Max Bucket Size: " << maxBucketSize << '\n';
}

void statisticsOfCustomHashTable() {
  size_t usedBuckets = 0;
  size_t totalItems = 0;
  size_t maxBucketSize = 0;

  for (size_t i = 0; i < N; i++) {
    size_t bucketSize = internalList[i].size();
    if (bucketSize > 0) {
      usedBuckets++;
      totalItems += bucketSize;
      if (bucketSize > maxBucketSize) {
        maxBucketSize = bucketSize;
      }
    }
  }

  double loadFactor = static_cast<double>(totalItems) / N;
  double usedSlots = static_cast<double>(usedBuckets) / N;

  std::cout << "Total Buckets: " << N << std::endl;
  std::cout << "Used Buckets: " << usedBuckets << std::endl;
  std::cout << "Total Items: " << totalItems << std::endl;
  std::cout << "Used Slots: " << usedSlots << std::endl;
  std::cout << "Load Factor: " << loadFactor << std::endl;
  std::cout << "Max Bucket Size: " << maxBucketSize << std::endl;

  size_t totalMemory = N * sizeof(std::vector<KeyValue<K, V>>);
  for (size_t i = 0; i < N; i++) {
    totalMemory += internalList[i].capacity() * sizeof(KeyValue<K, V>);
  }
  std::cout << "Total Memory Used (bytes): " << totalMemory << std::endl;
}
```
