#include <iostream>
#include <omp.h>
#include <random>

using namespace std;

int main() {
 int n = 100000000; // number of points to generate
    int counter = 0; // counter for points lying in the first quadrant of a unit circle
    auto start_time = omp_get_wtime(); // omp_get_wtime() is an OpenMP library routine

    #pragma omp parallel
    {
        // init random generator with thread number as seed
        unsigned int seed = omp_get_thread_num();
        default_random_engine re{seed};
        uniform_real_distribution<double> zero_to_one{0.0, 1.0};

        // local counter for each threadi
        int local_counter = 0;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            auto x = zero_to_one(re); 
            auto y = zero_to_one(re); 
            if (x * x + y * y <= 1.0) { 
                ++local_counter;
            }
        }

        
        #pragma omp atomic
        counter += local_counter;
    }

    auto run_time = omp_get_wtime() - start_time;
    auto pi = 4 * (double(counter) / n);

    cout << "pi: " << pi << endl;
    cout << "run_time: " << run_time << " s" << endl;
    cout << "n: " << n << endl; 
}
