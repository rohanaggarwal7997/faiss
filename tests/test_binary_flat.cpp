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
#include <faiss/index_io.h>
#include <stdio.h>

using namespace std;


TEST(BinaryFlat, accuracy) {
    // dimension of the vectors to index
    int d = 768;

    int ef_search_array[] = {128};

    const char* index_file_path = "/Users/rohaagga/Desktop/faiss_expts_python/SIFT_1M_SAVE_HNSW_EF40_M_32";
    const char* qfilename = "/Users/rohaagga/Desktop/faiss_expts_python/sift/sift_query.fvecs";

    // Read the index from the file
    faiss::Index* index = faiss::read_index(index_file_path);
    faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index);

    hnsw_index->hnsw.search_bounded_queue = 1;


    // Now the index is loaded in memory and you can use it for search, etc.

    hnsw_index->pretty_print2();
    cout<<"Number of elements "<<hnsw_index->ntotal;

    FILE* file = fopen(qfilename, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", qfilename);
        return;
    }

    int query_dimension;
    fread(&query_dimension, sizeof(int), 1, file);

    // Find the number of queries
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    int nq = file_size / (query_dimension * sizeof(float) + sizeof(int));
    fseek(file, sizeof(int), SEEK_SET);

    // Allocate memory for queries
    float* queries = (float*)malloc(nq * query_dimension * sizeof(float));
    if (!queries) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return;
    }

    // Read the vectors into queries
    for (int i = 0; i < nq; i++) {
        fread(queries + i * query_dimension, sizeof(float), query_dimension, file);
        fseek(file, sizeof(int), SEEK_CUR); // Skip the dimension value for the next vector
    }

    fclose(file);


    for (int p = 0; p < 1; p++)
    { // searching the database

        int ef_search = ef_search_array[p];

        faiss::SearchParametersHNSW search_params{};
        search_params.efSearch = ef_search;
        cout<<"Search paramaters "<<search_params.efSearch << " "
        << search_params.check_relative_distance <<endl;

        int k = 100;
        size_t sum_comparisions = 0;
        

        std::vector<faiss::idx_t> nns(k * 1);
        std::vector<float> dis(k * 1);

        for(int qq = 0; qq < 100; qq++)
        {
          index->search(1, queries + qq * query_dimension,
                        k, dis.data(), nns.data(), &search_params);
          sum_comparisions += hnsw_index->hnsw.get_total_comp();
        }
        cout<<"Total comparisions "<<sum_comparisions<<endl;

    }

    // Don't forget to delete the index to free the memory
    // Don't forget to free the memory when you are done
    free(queries);
    delete index;
}
