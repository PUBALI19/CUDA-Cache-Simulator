#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cmath>
#include <tuple>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include "cache_sim_flat.h"
using namespace std;
#include <cstring>

int main(int argc, char *argv[])
{
	FILE *fp;
	char *trace_file;
	char rw;
	uint32_t addr;

	if (argc != 2)
	{
	    printf("Error: Expected 1 command-line argument (trace file) but was provided %d.\n", (argc - 1));
	    exit(EXIT_FAILURE);
	}

	trace_file = argv[1];
	fp = fopen(trace_file, "r");
	if (fp == (FILE *)NULL) {
	    printf("Error: Unable to open file %s\n", trace_file);
	    exit(EXIT_FAILURE);
	}

	int total_no_of_ref = 0;
	{
	    char tmp_rw;
	    uint32_t tmp_addr;
	    while (fscanf(fp, "%c %x\n", &tmp_rw, &tmp_addr) == 2)
	    {
	        total_no_of_ref++;
	    }
	}
	rewind(fp);
	char* trace_rw = new char[total_no_of_ref];
	uint32_t* trace_addr = new uint32_t[total_no_of_ref];
	int idx = 0;
	while (fscanf(fp, "%c %x\n", &rw, &addr) == 2)
	{
	    trace_rw[idx] = rw;
	    trace_addr[idx] = addr;
	    idx++;
	}

	printf("===== Simulator configuration =====\n");
	printf("trace_file: %s\n\n", trace_file);

	int assoc_values[6] = {1, 2, 4, 8, 16, 32};
	int size_values[6] = {1024, 2048, 4096, 8192, 16384, 32768};
	int blocksize_values[3] = {16, 32, 64};
	int pref_n_values[3] = {0, 4, 8};
	int pref_m_values[2] = {2, 4};
	int num_assoc = 6;
	int num_size = 6;
	int num_blocksize = 3;
	int num_pref_n = 3;
	int num_pref_m = 2;
	int num_configs = num_assoc * num_size * num_blocksize * num_pref_n * num_pref_m;  // 648

	auto t0 = std::chrono::high_resolution_clock::now();

	for (int cfg = 0; cfg < num_configs; ++cfg)
	{
	    int pref_m_idx    = cfg % num_pref_m;
	    int pref_n_idx    = (cfg / num_pref_m) % num_pref_n;
	    int blocksize_idx = (cfg / (num_pref_m * num_pref_n)) % num_blocksize;
	    int size_idx      = (cfg / (num_pref_m * num_pref_n * num_blocksize)) % num_size;
	    int assoc_idx     = cfg / (num_pref_m * num_pref_n * num_blocksize * num_size);

	    uint32_t this_assoc     = assoc_values[assoc_idx];
	    uint32_t this_size      = size_values[size_idx];
	    uint32_t this_blocksize = blocksize_values[blocksize_idx];
	    uint32_t this_pref_n    = pref_n_values[pref_n_idx];
	    uint32_t this_pref_m    = pref_m_values[pref_m_idx];

	    if (this_size < this_blocksize * this_assoc) {
	        printf("ASSOC=%2u SIZE=%6u BLK=%2u PREF_N=%u PREF_M=%u -> skipped (invalid: fewer than 1 set)\n",
	               this_assoc, this_size, this_blocksize, this_pref_n, this_pref_m);
	        continue;
	    }

	    int l1_no_of_sets = this_size / (this_blocksize * this_assoc);

	    int* l1_arr_valid = new int[l1_no_of_sets * this_assoc];
	    memset(l1_arr_valid, 0, l1_no_of_sets * this_assoc * sizeof(int));
	    char* l1_arr_dirty = new char[l1_no_of_sets * this_assoc];
	    memset(l1_arr_dirty, ' ', l1_no_of_sets * this_assoc * sizeof(char));
	    int* l1_tag_storage = new int[l1_no_of_sets * this_assoc];
	    memset(l1_tag_storage, 0, l1_no_of_sets * this_assoc * sizeof(int));
	    uint32_t* l1_arr_dirty_addr = new uint32_t[l1_no_of_sets * this_assoc];
	    memset(l1_arr_dirty_addr, 0, l1_no_of_sets * this_assoc * sizeof(uint32_t));
	    int* l1_arr_lru = new int[l1_no_of_sets * this_assoc];
	    for (int i = 0; i < l1_no_of_sets; ++i)
	        for (uint32_t j = 0; j < this_assoc; ++j)
	            l1_arr_lru[i * this_assoc + j] = j;

	    // per-config stream buffer arrays (only meaningful if this_pref_n > 0)
	    bool* sb_valid = nullptr;
	    int* sb_top = nullptr;
	    int* sb_rec = nullptr;
	    uint32_t* sb_blocks = nullptr;
	    if (this_pref_n > 0) {
	        sb_valid = new bool[this_pref_n];
	        memset(sb_valid, false, this_pref_n * sizeof(bool));
	        sb_top = new int[this_pref_n];
	        memset(sb_top, 0, this_pref_n * sizeof(int));
	        sb_rec = new int[this_pref_n];
	        for (uint32_t i = 0; i < this_pref_n; ++i) sb_rec[i] = this_pref_n - i;
	        sb_blocks = new uint32_t[this_pref_n * this_pref_m];
	        memset(sb_blocks, 0, this_pref_n * this_pref_m * sizeof(uint32_t));
	    }

	    cache_params l1_counters;
	    cache_params l2_counters;

	    for (int i = 0; i < total_no_of_ref; ++i)
	    {
	        l1_cache(l1_tag_storage, l1_arr_dirty, l1_arr_valid, l1_arr_dirty_addr, l1_arr_lru,
	                 nullptr, nullptr, nullptr, nullptr, nullptr,
	                 l1_counters, l2_counters, this_blocksize, this_size, this_assoc,
	                 trace_addr[i], trace_rw[i], 0, 0,
	                 sb_valid, sb_top, sb_rec, sb_blocks,
	                 this_pref_n, this_pref_m, 0, 0);
	    }

	    double miss_rate = static_cast<double>(l1_counters.write_miss + l1_counters.read_miss)
	        / (l1_counters.read_hit + l1_counters.write_hit + l1_counters.read_miss + l1_counters.write_miss
	           + l1_counters.sb_read_hits + l1_counters.sb_write_hits);

	    printf("ASSOC=%2u SIZE=%6u BLK=%2u PREF_N=%u PREF_M=%u -> L1 miss rate = %.4f\n",
	           this_assoc, this_size, this_blocksize, this_pref_n, this_pref_m, miss_rate);

	    delete[] l1_arr_valid;
	    delete[] l1_arr_dirty;
	    delete[] l1_tag_storage;
	    delete[] l1_arr_dirty_addr;
	    delete[] l1_arr_lru;
	    if (sb_valid) delete[] sb_valid;
	    if (sb_top) delete[] sb_top;
	    if (sb_rec) delete[] sb_rec;
	    if (sb_blocks) delete[] sb_blocks;
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	printf("\nCPU sweep time: %.3f ms\n", cpu_ms);

	delete[] trace_rw;
	delete[] trace_addr;

	return 0;
}