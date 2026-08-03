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
#include "cache_sim_gpu.h"
using namespace std;
#include <cstring>

__global__ void sweep_kernel(int num_configs, int num_size, int num_blocksize, int num_pref_n, int num_pref_m,
                              int max_sets, int max_assoc, int max_pref_n, int max_pref_m,
                              uint32_t* assoc_values_arr, uint32_t* size_values_arr, uint32_t* blocksize_values_arr,
                              uint32_t* pref_n_values_arr, uint32_t* pref_m_values_arr,
                              int* l1_arr_valid, int* l1_tag_storage, char* l1_arr_dirty,
                              uint32_t* l1_arr_dirty_addr, int* l1_arr_lru,
                              bool* sb_valid_g, int* sb_top_g, int* sb_rec_g, uint32_t* sb_blocks_g,
                              cache_params* counters,
                              char* trace_rw, uint32_t* trace_addr, int total_no_of_ref)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_configs) return;

    int pref_m_idx    = tid % num_pref_m;
    int pref_n_idx    = (tid / num_pref_m) % num_pref_n;
    int blocksize_idx = (tid / (num_pref_m * num_pref_n)) % num_blocksize;
    int size_idx      = (tid / (num_pref_m * num_pref_n * num_blocksize)) % num_size;
    int assoc_idx     = tid / (num_pref_m * num_pref_n * num_blocksize * num_size);

    uint32_t this_assoc     = assoc_values_arr[assoc_idx];
    uint32_t this_size      = size_values_arr[size_idx];
    uint32_t this_blocksize = blocksize_values_arr[blocksize_idx];
    uint32_t this_pref_n    = pref_n_values_arr[pref_n_idx];
    uint32_t this_pref_m    = pref_m_values_arr[pref_m_idx];

    int this_no_of_sets = this_size / (this_blocksize * this_assoc);
    size_t base           = (size_t)tid * max_sets * max_assoc;
    size_t sb_base         = (size_t)tid * max_pref_n;
    size_t sb_blocks_base  = (size_t)tid * max_pref_n * max_pref_m;

    int* my_valid       = l1_arr_valid + base;
    int* my_tags        = l1_tag_storage + base;
    char* my_dirty       = l1_arr_dirty + base;
    uint32_t* my_dirty_addr = l1_arr_dirty_addr + base;
    int* my_lru         = l1_arr_lru + base;

    bool* my_sb_valid      = sb_valid_g + sb_base;
    int* my_sb_top         = sb_top_g + sb_base;
    int* my_sb_rec         = sb_rec_g + sb_base;
    uint32_t* my_sb_blocks = sb_blocks_g + sb_blocks_base;

    cache_params my_counters;
    cache_params dummy_l2_counters;

    if (this_no_of_sets == 0) {
        counters[tid] = my_counters;
        return;
    }

    for (int s = 0; s < this_no_of_sets; ++s)
        for (uint32_t w = 0; w < this_assoc; ++w)
            my_lru[s * this_assoc + w] = w;

    for (uint32_t i = 0; i < this_pref_n; ++i) {
        my_sb_valid[i] = false;
        my_sb_top[i] = 0;
        my_sb_rec[i] = this_pref_n - i;
    }
    for (uint32_t i = 0; i < this_pref_n * this_pref_m; ++i) {
        my_sb_blocks[i] = 0;
    }

    for (int i = 0; i < total_no_of_ref; ++i)
    {
        l1_cache(my_tags, my_dirty, my_valid, my_dirty_addr, my_lru,
                 nullptr, nullptr, nullptr, nullptr, nullptr,
                 my_counters, dummy_l2_counters,
                 this_blocksize, this_size, this_assoc,
                 trace_addr[i], trace_rw[i], 0, 0,
                 my_sb_valid, my_sb_top, my_sb_rec, my_sb_blocks,
                 this_pref_n, this_pref_m, 0, 0);
    }

    counters[tid] = my_counters;
}

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
	const int num_configs = num_assoc * num_size * num_blocksize * num_pref_n * num_pref_m;  // 648

	int max_assoc = 32;
	int max_size = 32768;
	int min_blocksize = 16;
	int max_sets = max_size / (min_blocksize * 1);
	int max_pref_n = 8;
	int max_pref_m = 4;

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

	int *g_l1_arr_valid, *g_l1_tag_storage, *g_l1_arr_lru;
	char *g_l1_arr_dirty;
	uint32_t *g_l1_arr_dirty_addr;
	cache_params *g_counters;
	uint32_t *g_assoc_values, *g_size_values, *g_blocksize_values, *g_pref_n_values, *g_pref_m_values;
	char *g_trace_rw;
	uint32_t *g_trace_addr;
	bool *g_sb_valid;
	int *g_sb_top, *g_sb_rec;
	uint32_t *g_sb_blocks;

	size_t per_thread_size = max_sets * max_assoc;

	cudaMallocManaged(&g_assoc_values, num_assoc * sizeof(uint32_t));
	cudaMallocManaged(&g_size_values, num_size * sizeof(uint32_t));
	cudaMallocManaged(&g_blocksize_values, num_blocksize * sizeof(uint32_t));
	cudaMallocManaged(&g_pref_n_values, num_pref_n * sizeof(uint32_t));
	cudaMallocManaged(&g_pref_m_values, num_pref_m * sizeof(uint32_t));
	for (int i = 0; i < num_assoc; ++i) g_assoc_values[i] = assoc_values[i];
	for (int i = 0; i < num_size; ++i) g_size_values[i] = size_values[i];
	for (int i = 0; i < num_blocksize; ++i) g_blocksize_values[i] = blocksize_values[i];
	for (int i = 0; i < num_pref_n; ++i) g_pref_n_values[i] = pref_n_values[i];
	for (int i = 0; i < num_pref_m; ++i) g_pref_m_values[i] = pref_m_values[i];

	cudaMallocManaged(&g_l1_arr_valid,      (size_t)num_configs * per_thread_size * sizeof(int));
	cudaMallocManaged(&g_l1_tag_storage,    (size_t)num_configs * per_thread_size * sizeof(int));
	cudaMallocManaged(&g_l1_arr_dirty,      (size_t)num_configs * per_thread_size * sizeof(char));
	cudaMallocManaged(&g_l1_arr_dirty_addr, (size_t)num_configs * per_thread_size * sizeof(uint32_t));
	cudaMallocManaged(&g_l1_arr_lru,        (size_t)num_configs * per_thread_size * sizeof(int));
	cudaMallocManaged(&g_counters,          num_configs * sizeof(cache_params));
	cudaMallocManaged(&g_trace_rw,          total_no_of_ref * sizeof(char));
	cudaMallocManaged(&g_trace_addr,        total_no_of_ref * sizeof(uint32_t));

	cudaMallocManaged(&g_sb_valid,  (size_t)num_configs * max_pref_n * sizeof(bool));
	cudaMallocManaged(&g_sb_top,    (size_t)num_configs * max_pref_n * sizeof(int));
	cudaMallocManaged(&g_sb_rec,    (size_t)num_configs * max_pref_n * sizeof(int));
	cudaMallocManaged(&g_sb_blocks, (size_t)num_configs * max_pref_n * max_pref_m * sizeof(uint32_t));

	for (int i = 0; i < total_no_of_ref; ++i) { g_trace_rw[i] = trace_rw[i]; g_trace_addr[i] = trace_addr[i]; }

	memset(g_l1_arr_valid, 0, (size_t)num_configs * per_thread_size * sizeof(int));
	memset(g_l1_arr_dirty, ' ', (size_t)num_configs * per_thread_size * sizeof(char));
	memset(g_l1_arr_dirty_addr, 0, (size_t)num_configs * per_thread_size * sizeof(uint32_t));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int threadsPerBlock = 32;
	int numBlocks = (num_configs + threadsPerBlock - 1) / threadsPerBlock;

	cudaEventRecord(start);
	sweep_kernel<<<numBlocks, threadsPerBlock>>>(num_configs, num_size, num_blocksize, num_pref_n, num_pref_m,
	                                  max_sets, max_assoc, max_pref_n, max_pref_m,
	                                  g_assoc_values, g_size_values, g_blocksize_values,
	                                  g_pref_n_values, g_pref_m_values,
	                                  g_l1_arr_valid, g_l1_tag_storage, g_l1_arr_dirty,
	                                  g_l1_arr_dirty_addr, g_l1_arr_lru,
	                                  g_sb_valid, g_sb_top, g_sb_rec, g_sb_blocks,
	                                  g_counters,
	                                  g_trace_rw, g_trace_addr, total_no_of_ref);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float kernel_ms = 0;
	cudaEventElapsedTime(&kernel_ms, start, stop);
	printf("Kernel execution time: %.3f ms\n", kernel_ms);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	for (int i = 0; i < num_configs; ++i) {
	    cache_params &c = g_counters[i];
	    int pref_m_idx    = i % num_pref_m;
	    int pref_n_idx    = (i / num_pref_m) % num_pref_n;
	    int blocksize_idx = (i / (num_pref_m * num_pref_n)) % num_blocksize;
	    int size_idx      = (i / (num_pref_m * num_pref_n * num_blocksize)) % num_size;
	    int assoc_idx     = i / (num_pref_m * num_pref_n * num_blocksize * num_size);

	    uint32_t a  = g_assoc_values[assoc_idx];
	    uint32_t s  = g_size_values[size_idx];
	    uint32_t b  = g_blocksize_values[blocksize_idx];
	    uint32_t pn = g_pref_n_values[pref_n_idx];
	    uint32_t pm = g_pref_m_values[pref_m_idx];

	    int total = c.read_hit + c.write_hit + c.read_miss + c.write_miss + c.sb_read_hits + c.sb_write_hits;
	    if (total == 0) {
	        printf("ASSOC=%2u SIZE=%6u BLK=%2u PREF_N=%u PREF_M=%u -> skipped (invalid: fewer than 1 set)\n", a, s, b, pn, pm);
	    } else {
	        double miss_rate = static_cast<double>(c.write_miss + c.read_miss) / total;
	        printf("ASSOC=%2u SIZE=%6u BLK=%2u PREF_N=%u PREF_M=%u -> L1 miss rate = %.4f\n", a, s, b, pn, pm, miss_rate);
	    }
	}

	delete[] trace_rw;
	delete[] trace_addr;
	cudaFree(g_l1_arr_valid);
	cudaFree(g_l1_tag_storage);
	cudaFree(g_l1_arr_dirty);
	cudaFree(g_l1_arr_dirty_addr);
	cudaFree(g_l1_arr_lru);
	cudaFree(g_counters);
	cudaFree(g_assoc_values);
	cudaFree(g_size_values);
	cudaFree(g_blocksize_values);
	cudaFree(g_pref_n_values);
	cudaFree(g_pref_m_values);
	cudaFree(g_trace_rw);
	cudaFree(g_trace_addr);
	cudaFree(g_sb_valid);
	cudaFree(g_sb_top);
	cudaFree(g_sb_rec);
	cudaFree(g_sb_blocks);

	return 0;
}