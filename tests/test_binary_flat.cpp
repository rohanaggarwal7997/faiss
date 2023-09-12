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
#include <faiss/impl/IDSelector.h>
#include <vector>
#include <cstdlib>
#include <unordered_set>

using namespace std;

faiss::IDSelectorBatch createRandomSelector(int total_vids, vector<faiss::idx_t> rejected) {
    std::vector<faiss::idx_t> ids;
    std::unordered_set<faiss::idx_t> rejectedSet(rejected.begin(), rejected.end());

    for(int i = 1; i <= total_vids; ++i) {
        if(rejectedSet.find(i) == rejectedSet.end()) {
            ids.push_back(i);
        }
    }
    return faiss::IDSelectorBatch(ids.size(), ids.data());
}

TEST(BinaryFlat, accuracy) {


    std::cout << "RAND_MAX: " << RAND_MAX << std::endl;


    vector<vector<faiss::idx_t>> rejected_array;

    for(int i =0; i < 100; i++)
    {
        vector<faiss::idx_t> r;
        rejected_array.push_back(r);
    }

    char* index_file_path_array[] = 
    {"/Users/rohaagga/Desktop/faiss_expts_python/SIFT_1M_SAVE_HNSW_EF40_M_32",
    "/Users/rohaagga/Desktop/faiss_expts_python/SIFT_500K_SAVE_HNSW_EF40_M_32",
    "/Users/rohaagga/Desktop/faiss_expts_python/SIFT_100K_SAVE_HNSW_EF40_M_32"
    }
    ;
    int total_vids_array[] = 
    {1000000, 500000, 100000
    }
    ;
    const char* qfilename = "/Users/rohaagga/Desktop/faiss_expts_python/sift/sift_query.fvecs";

    int total_vids;
    char* index_file_path;


    float percentage_array[] = {0.001,0.01, 0.1,0.3,0.5,0.7,0.9};

    for (int index_array = 0; index_array < 3; index_array++)
    {
        index_file_path = index_file_path_array[index_array];
        total_vids = total_vids_array[index_array];

        for(int i =0; i < 100; i++)
        {
            rejected_array[i].clear();
        }

    for (int perp = 0; perp < 10; perp++)
    {

    // dimension of the vectors to index
    int d = 768;

    int ef_search_array[] = {128};



    // Read the index from the file
    faiss::Index* index = faiss::read_index(index_file_path);
    faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index);

    hnsw_index->hnsw.search_bounded_queue = 0;


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
          faiss::IDSelectorBatch selector = createRandomSelector(total_vids, rejected_array[qq]);
          search_params.sel = &selector;

          if (qq == 0)
          {
              cout<<" rejected array size"<<rejected_array[qq].size()<<endl;
          }
          
          index->search(1, queries + qq * query_dimension,
                        k, dis.data(), nns.data(), &search_params);
          sum_comparisions += hnsw_index->hnsw.get_total_comp();
          std::set<faiss::idx_t> nns_set(nns.begin(), nns.end());
          
          vector<faiss::idx_t> current_rejected = hnsw_index->hnsw.get_path_vec();
          if (qq == 0)
          {
              cout<<"Current rejected size"<<current_rejected.size()<<endl;
          }
          for (int iii = 0; iii < current_rejected.size(); iii++)
          {
              if(nns_set.find(current_rejected[iii]) == nns_set.end())
              {
                rejected_array[qq].push_back(current_rejected[iii]);
              }
          }

        //   for(int i = 0; i < k; ++i) {
        //     std::cout << "Neighbor " << i << ": Index = " << nns[i] 
        //         << ", Distance = " << dis[i] << "\n";
        //   }
        //   std::cout << "\n"; // Print a blank line between queries
        }
        cout<<"Total comparisions "<<sum_comparisions<<endl;

    }

    // Don't forget to delete the index to free the memory
    // Don't forget to free the memory when you are done
    free(queries);
    delete index;

    }
    }
}
