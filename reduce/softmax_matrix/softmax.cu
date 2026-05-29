#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>


void softmax_matrix_cpu(const float* input, float* output, size_t N, size_t M){
    // matrix shape M x N
    for(int i=0; i<M; i++){
        float max_val = *(std::max_element(input + i * N, input + (i+1)* N));
        float row_sum = 0.0f;
        for(int j=0; j<N; j++){
            output[i*N + j] = expf(input[i*N + j] - max_val);
            row_sum += output[i*N + j];
        }
        for(int j=0; j<N; j++){
            output[i*N + j] /= row_sum;
        }
    }
}

void softmax_matrix_col_cpu(const float* input, float* output, size_t N, size_t M){
    // matrix shape M x N
    for(int i=0; i<N; i++){
        float max_val = -INFINITY;
        for(int j=0; j< M; j++){
            max_val = fmaxf(max_val, input[j*N + i]);
        }
        float col_sum = 0.0f;
        for(int j=0; j< M; j++){
            output[j*N + i] = expf(input[j*N + i] - max_val);
            col_sum += output[j*N + i];
        }
        for(int j=0; j< M; j++){
            output[j*N + i] /= col_sum;
        }
    }
}


