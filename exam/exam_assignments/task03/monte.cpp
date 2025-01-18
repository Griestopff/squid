#include <iostream>
#include <omp.h>
#include <random>

using namespace std;

// simple random number function
double rnd(unsigned int* seed) {
    *seed = (1140671485 * (*seed) + 12820163) % (1 << 24);
    return static_cast<double>(*seed) / (1 << 24);
}

int main() {
    int n = 100000000; 
    int counter = 0;   
    auto start_time = omp_get_wtime(); 

    #pragma omp parallel reduction(+:counter) // reduction counter -> no atomic region 
    {
        unsigned int seed = omp_get_thread_num(); // unique seed for each thread
        int local_counter = 0;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            double x = rnd(&seed); 
            double y = rnd(&seed); 
            if (x * x + y * y <= 1.0) {
                ++local_counter;
            }
        }
        counter += local_counter;
    }

    auto run_time = omp_get_wtime() - start_time;
    auto pi = 4.0 * static_cast<double>(counter) / n;

    cout << "pi: " << pi << endl;
    cout << "run_time: " << run_time << " s" << endl;
    cout << "n: " << n << endl;
}

