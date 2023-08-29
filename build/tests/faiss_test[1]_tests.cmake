add_test([=[BinaryFlat.accuracy]=]  /Users/rohaagga/Desktop/Faiss/faiss/build/tests/faiss_test [==[--gtest_filter=BinaryFlat.accuracy]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[BinaryFlat.accuracy]=]  PROPERTIES WORKING_DIRECTORY /Users/rohaagga/Desktop/Faiss/faiss/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  faiss_test_TESTS BinaryFlat.accuracy)
