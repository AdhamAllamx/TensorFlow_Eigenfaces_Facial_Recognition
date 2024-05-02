#include "pca_parameters.h"  // Ensure this file contains pca_mean_vector and pca_transformation_matrix definitions

void applyPCA(const float* input, float* output) {
    float centered[4096];
    for (int i = 0; i < 4096; i++) {
        centered[i] = input[i] - pca_mean_vector[i];
    }

    for (int i = 0; i < 120; i++) {
        output[i] = 0;
        for (int j = 0; j < 4096; j++) {
            output[i] += centered[j] * pca_transformation_matrix[i][j];
        }
    }
}
