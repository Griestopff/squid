// Kompilierungsbefehl: g++ min_max_quicksort.cpp -fopenmp -Ofast -march=native -ffast-math -o bench

#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <parallel/algorithm>
#include <fstream>

// Berechnet den Durchschnitt zweier Integer-Werte ohne Überlauf
inline int64_t average(int64_t a, int64_t b) {
    return (a & b) + ((a ^ b) >> 1);
}

// Partitionierungsfunktion für Quicksort
inline int64_t partition(int64_t *arr, int64_t left, int64_t right, int64_t pivot, int64_t &smallest, int64_t &biggest) {
    int64_t *left_ptr = &arr[left];
    int64_t *right_ptr = &arr[right];
    while (left_ptr < right_ptr) {
        smallest = (*left_ptr < smallest) ? *left_ptr : smallest;
        biggest = (*left_ptr > biggest) ? *left_ptr : biggest;
        if (*left_ptr > pivot) {
            --right_ptr;
            std::swap(*left_ptr, *right_ptr);
        } else {
            ++left_ptr;
        }
    }
    return left_ptr - arr;
}

inline void insertion_sort(int64_t *arr, int64_t left, int64_t right) {
    for (int64_t i = left + 1; i <= right; i++) {
        int64_t key = arr[i];
        int64_t j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// Rekursive Quicksort-Funktion
void qs_core(int64_t *arr, int64_t left, int64_t right, const int64_t pivot) {
    if (right - left < 32) {
        insertion_sort(arr, left, right);
        return;
    }

    int64_t smallest = std::numeric_limits<int64_t>::max();
    int64_t biggest = std::numeric_limits<int64_t>::min();
    int64_t bound = partition(arr, left, right + 1, pivot, smallest, biggest);

    if (smallest == biggest)
        return;

#pragma omp task final(bound - left < 10000)
    qs_core(arr, left, bound - 1, average(smallest, pivot));
    qs_core(arr, bound, right, average(pivot, biggest));
}

// Wrapper für die Quicksort-Funktion
void min_max_quicksort(int64_t *arr, int64_t n, int num_threads = omp_get_max_threads()) {
#pragma omp parallel num_threads(num_threads)
#pragma omp single nowait
    qs_core(arr, 0, n - 1, n > 0 ? arr[average(0, n - 1)] : 0);
}

// Klasse zur Generierung von Zufallszahlen
class Xoroshiro128Plus {
    uint64_t state[2]{};

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit Xoroshiro128Plus(uint64_t seed = 0) {
        state[0] = (12345678901234567 + seed) | 0b1001000010000001000100101000000110010010100000011001001010000001ULL;
        state[1] = (98765432109876543 + seed) | 0b0100000011001100100000011001001010000000100100101000000110010010ULL;
        for(int i = 0; i < 10; i++){operator()();}
    }

    uint64_t operator()() {
        const uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        state[1] = rotl(s1, 37);
        return result;
    }
};

int main() {
    const int64_t SIZE = 10000000;
    uint64_t seed = 3;
    Xoroshiro128Plus generator(seed);
    std::ofstream file("speedup_results_threads.csv");
    
    // CSV-Header
    file << "Threads,Time_min_max_quicksort,Speedup_min_max_quicksort,Time_gnu_parallel,Speedup_gnu_parallel\n";
    
    // Messungen für verschiedene Thread-Anzahlen
    for (int threads = 1; threads <= 16; ++threads) {
        double total_time_min_max_quicksort = 0;
        double total_time_gnu_parallel = 0;
        double total_time_std_sort = 0;
        
        // Führe den Test 5 Mal durch und berechne den Durchschnitt
        for (int i = 0; i < 5; ++i) {
            std::vector<int64_t> data(SIZE);
            for (int64_t i = 0; i < SIZE; ++i) {
                data[i] = generator();
            }

            // Kopien für verschiedene Sortieralgorithmen
            std::vector<int64_t> data_copy = data;
            std::vector<int64_t> data_copy_parallel = data;

            // std::sort messen
            double start = omp_get_wtime();
            std::sort(data.begin(), data.end());
            double end = omp_get_wtime();
            total_time_std_sort += (end - start);

            // min_max_quicksort messen
            omp_set_num_threads(threads);
            start = omp_get_wtime();
            min_max_quicksort(&data_copy[0], SIZE, threads);
            end = omp_get_wtime();
            total_time_min_max_quicksort += (end - start);

            // __gnu_parallel::sort messen
            start = omp_get_wtime();
            __gnu_parallel::sort(data_copy_parallel.begin(), data_copy_parallel.end());
            end = omp_get_wtime();
            total_time_gnu_parallel += (end - start);
        }

        // Durchschnittswerte berechnen
        double avg_time_min_max_quicksort = total_time_min_max_quicksort / 5.0;
        double avg_time_gnu_parallel = total_time_gnu_parallel / 5.0;
        double avg_time_std_sort = total_time_std_sort / 5.0;

        // Speedups berechnen
        double speedup_min_max = avg_time_std_sort / avg_time_min_max_quicksort;
        double speedup_gnu_parallel = avg_time_std_sort / avg_time_gnu_parallel;

        // Ergebnisse in die CSV-Datei schreiben
        file << threads << "," << avg_time_min_max_quicksort << "," << speedup_min_max << ","
             << avg_time_gnu_parallel << "," << speedup_gnu_parallel << "\n";
        
        std::cout << "Threads: " << threads 
                  << " | min_max_quicksort time: " << avg_time_min_max_quicksort 
                  << " s, Speedup: " << speedup_min_max
                  << " | __gnu_parallel::sort time: " << avg_time_gnu_parallel 
                  << " s, Speedup: " << speedup_gnu_parallel << "\n";
    }

    file.close();
    std::cout << "Ergebnisse wurden in 'speedup_results_threads.csv' gespeichert.\n";
    return 0;
}

