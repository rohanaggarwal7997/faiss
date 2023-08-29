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
    // dimension of the vectors to index
    int d = 16;

    int m_array[] = {32,48, 64};
    int ef_construction[] = {40,60,100};
    int size_array[] = {10000,100000,500000,1000000};
    int ef_search_array[] = {32,48,64,96,128,256};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // size of the database we plan to index
            size_t nb = size_array[j];

            // make the index object and train it
            faiss::IndexHNSWFlat index(d, m_array[i]);

            index.hnsw.efConstruction = ef_construction[i];

            index.pretty_print2();
            cout<<"Number of elements "<<nb<<endl;

            float *database = new float[nb*d];
            for (size_t i = 0; i < nb * (d); i++) {
                database[i] = (float)(rand() % 0x100);
            }

            { // populating the database
                index.add(nb, database);
            }

            size_t nq = 1;
            float *queries = new float[nq*d];

            for (size_t i = 0; i < nq * (d); i++) {
                queries[i] = (float)(rand() % 0x100);
            }

            for (int p = 0; p < 6; p++)
            { // searching the database

                int ef_search = ef_search_array[p];

                faiss::SearchParametersHNSW search_params{};
                search_params.efSearch = ef_search;
                cout<<"Search paramaters "<<search_params.efSearch << " "
                << search_params.check_relative_distance <<endl;

                int k = 10;
                

                std::vector<faiss::idx_t> nns(k * nq);
                std::vector<float> dis(k * nq);

                index.search(nq, queries, k, dis.data(), nns.data(), &search_params);
            }

        }

    }
}
