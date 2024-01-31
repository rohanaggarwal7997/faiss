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


TEST(BinaryFlat, accuracy) {

    omp_set_num_threads(32);

    // dimension of the vectors to index
    int d = 960;

    int m_array[] = {32};
    int ef_construction[] = {64,200};
    int size_array[] = {1000000};

    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            // size of the database we plan to index
            size_t nb = size_array[0];

            // make the index object and train it
            faiss::IndexHNSWFlat index(d, m_array[i]);

            index.hnsw.efConstruction = ef_construction[j];

            cout<<"Rohan i am here"<<endl;

            index.hnsw.set_total_comp(0);

            cout<<"Rohan i am here"<<endl;

            float *database = new float[nb*d];
            for (size_t i = 0; i < nb * (d); i++) {
                database[i] = (float)(rand() % 0x100);
            }

            cout<<"Rohan i am here"<<endl;

            { // populating the database
                index.add(nb, database);
            }

            cout<<"Rohan i am here"<<endl;

            cout<<"Rohan experiment"<<index.hnsw.get_total_comp();

        }

    }
}
