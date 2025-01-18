#include <complex>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>

using namespace std;

constexpr int width = 4000;
constexpr int height = width;

int compute_pixel(int x, int y) { // compute the gray value of a pixel
    complex<double> point(2.0 * x / width - 1.5, 2.0 * y / height - 1.0);
    complex<double> z(0, 0);
    constexpr int iterations = 100;
    int nb_iter = 0;
    while (abs(z) < 2 && nb_iter < iterations) {
        z = z * z + point;
        nb_iter++;
    }
    return (255 * nb_iter) / iterations;
}

int main() { 

    const string image_name = "mandelbrot.pgm";
    remove(image_name.c_str());
    const double start = omp_get_wtime();
    ofstream image(image_name); 
    
    if (image.is_open()) {
        image << "P2\n" << width << " " << height << " 255\n"; // pgm header (only one)

        
        #pragma omp parallel for schedule(static, 1) ordered // parallelizing the for loop
        for (int i = 0; i < height; i++) {
            // thread-private stringstream to accumulate pixel data
            stringstream local_image_stream;

            for (int j = 0; j < width; j++) {
                local_image_stream << compute_pixel(j, i) << "\n"; 
            }

            // output the accumulated data from the thread-local stringstream
            #pragma omp ordered // ensure that only one thread writes to the file at a time and in correct order of i        
                image << local_image_stream.str();
            
        }

        image.close(); 
    } else {
        cout << "Could not open the file!";
    }
    cout << omp_get_wtime() - start << " seconds" << endl;
}

