/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/hamming.h>
using namespace std;

#include <iostream>
#include <fstream>
#include <vector>


// Function to read fvecs file
void read_fvecs(const char *filename, float *&data, size_t &num, size_t &dim) {
    ifstream input(filename, ios::binary);
    if (!input.is_open()) {
        cerr << "Could not open the file " << filename << endl;
        exit(1);
    }

    // Read the dimension of the first vector to set the dimensionality
    int d;
    input.read((char *)&d, sizeof(int));
    input.seekg(0, ios::end);
    size_t fileSize = input.tellg();
    // Reset to start
    input.seekg(0, ios::beg);

    // Calculate the number of vectors
    num = fileSize / (4 + d * 4); // Each vector has an integer for dimension and d floats
    dim = d;

    data = new float[num * dim];

    for (size_t i = 0; i < num; i++) {
        // Skip the dimension part for each vector
        input.seekg(4, ios::cur);
        input.read((char *)(data + i * dim), dim * sizeof(float));
    }

    input.close();
}

TEST(BinaryFlat, accuracy) {
    omp_set_num_threads(32);

    // dimension of the vectors to index
    int d = 960;

    int m_array[] = {32};
    int ef_construction[] = {64, 200};
    int size_array[] = {1000000};

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            // size of the database we plan to index
            size_t nb = size_array[0];

            // make the index object and train it
            faiss::IndexHNSWFlat index(d, m_array[i]);
            index.hnsw.efConstruction = ef_construction[j];

            cout << "Rohan i am here" << endl;

            index.hnsw.set_total_comp(0);

            cout << "Rohan i am here" << endl;

            float *database = nullptr;
            size_t num, dim;
            // Replace "/path/to/your/dataset.fvecs" with the actual file path
            read_fvecs("/Users/rohaagga/Desktop/gist_base.fvecs", database, num, dim);

            if (dim != d) {
                cerr << "Dimensionality of the dataset does not match the index dimension." << endl;
                exit(1);
            }

            cout << "Rohan i am here" << endl;

            // Populating the database
            index.add(nb, database);

            cout << "Rohan i am here" << endl;

            cout << "Rohan experiment" << index.hnsw.get_total_comp();

            // Don't forget to free the memory
            delete[] database;
        }
    }
}