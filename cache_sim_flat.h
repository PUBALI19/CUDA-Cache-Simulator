#ifndef SIM_CACHE_FLAT_H
#define SIM_CACHE_FLAT_H
#include <cmath>
#include <tuple>
#include <utility>
#include <cstdint>
#include <climits>

typedef
struct {
   uint32_t BLOCKSIZE;
   uint32_t L1_SIZE;
   uint32_t L1_ASSOC;
   uint32_t L2_SIZE;
   uint32_t L2_ASSOC;
  uint32_t PREF_N;
  uint32_t PREF_M;
} cache_params_t;


//structure for all the cache parameters
struct cache_params
{
    int read_hit = 0;
    int write_hit = 0;
    int read_miss = 0;
    int write_miss = 0;
    int writeback_to_lower_mem = 0;

    int prefetch_reads_from_l1 = 0;
    int prefetch_misses_from_l1 = 0;

    int prefetch_req = 0;  //prefetches issued

    //hits on stream buffer
    int sb_read_hits = 0;
    int sb_write_hits = 0;
};



	//for implemetation of lru policy wherein counters are maintained indicating the recency
	//0: most recently used
	void lru_policy(int l_assoc, int l_index_set, int l_index_tag, int* l_arr_lru)
  {
    for (int q = 0; q < l_assoc; ++q) {
        if (l_arr_lru[l_index_set * l_assoc + q] < l_arr_lru[l_index_set * l_assoc + l_index_tag])
            l_arr_lru[l_index_set * l_assoc + q]++;
    }
    l_arr_lru[l_index_set * l_assoc + l_index_tag] = 0;
  }


  //getting the number of sets and number of bits required for set and blockoffset
	std::tuple<int, int, int> set_tag_bits_cal(int block_size, int cache_size, int assoc)
	{
	    int no_of_sets = cache_size / (block_size * assoc);
	    int set_bits = std::log2(no_of_sets);
	    int blockoffset_bits = std::log2(block_size);
	    return {set_bits, blockoffset_bits, no_of_sets};
	}
  //parsing out the set and tag fields out of the 32 bits address
	std::pair<uint32_t, uint32_t> parseaddress(uint32_t address, int no_set_bits, int no_blockoffset_bits)
 	{
	    uint32_t index_mask = (1u << no_set_bits) - 1;
	    uint32_t set = (address >> no_blockoffset_bits) & index_mask;
	    uint32_t tag = address >> (no_blockoffset_bits + no_set_bits);
	    return {tag, set};
	}

  void sb_mark_mru(int p_index, int* sb_rec, uint32_t pref_N)
  {
    if (pref_N == 0)
    return;
    for (size_t i = 0; i < pref_N; ++i) {
        if ((int)i != p_index)
            sb_rec[i] += 1;
    }
    sb_rec[p_index] = 0;
  }
  int sb_hit(uint32_t pref_N, uint32_t pref_M, bool* sb_valid, int* sb_top, int* sb_rec, uint32_t* sb_blocks, uint32_t block_number)
  {
    if (pref_N == 0) return -1;
    int selected = -1;
    int best_rec = INT32_MAX;
    for (size_t i = 0; i < pref_N; ++i) {
        if (!sb_valid[i]) continue;
        for (uint32_t j = 0; j < pref_M; ++j) {
            uint32_t real_index = (sb_top[i] + j) % pref_M;
            if (sb_blocks[i*pref_M + real_index] == block_number) {
                if (sb_rec[i] < best_rec) {
                    selected = (int)i;
                    best_rec = sb_rec[i];
                }
                break;
            }
        }
    }
    return selected;
  }

  int sb_pos_finder(int buf_p_index, uint32_t block_number, uint32_t pref_N, uint32_t pref_M, int* sb_top, uint32_t* sb_blocks)
  {
    if (buf_p_index < 0 || (uint32_t)buf_p_index >= pref_N) return -1;
    for (uint32_t j = 0; j < pref_M; ++j) {
        uint32_t real_index = (sb_top[buf_p_index] + j) % pref_M;
        if (sb_blocks[buf_p_index*pref_M+real_index] == block_number)
            return (int)j;
    }
    return -1;
  }

  void sb_miss(int buf_p_index, uint32_t start_block, bool* sb_valid, int* sb_top, uint32_t* sb_blocks, uint32_t pref_M, uint32_t pref_N, int* sb_rec)
  {
    if (pref_N == 0) return;
    sb_valid[buf_p_index] = true;
    sb_top[buf_p_index] = 0;
    uint32_t next = start_block + 1;
    for (uint32_t i = 0; i < pref_M; ++i)
        sb_blocks[buf_p_index*pref_M + i] = next + i;
    sb_mark_mru(buf_p_index, sb_rec, pref_N);
  }
  void sb_hit_continue(int buf_p_index, int pos, uint32_t pref_M, uint32_t pref_N,
                     int* sb_top, uint32_t* sb_blocks, int* sb_rec)
  {
    if (pref_N == 0) return;
    int count =0;
    uint32_t* remaining = new uint32_t[pref_M];
    for (uint32_t i = pos + 1; i < pref_M; ++i) {
        uint32_t real_index = (sb_top[buf_p_index] + i) % pref_M;
        remaining[count++] = sb_blocks[buf_p_index * pref_M + real_index];
    }
    uint32_t last_elem_index = (sb_top[buf_p_index]+ pref_M - 1) % pref_M;
    uint32_t next_prefetch = sb_blocks[buf_p_index * pref_M + last_elem_index] + 1;
    uint32_t* newblocks = new uint32_t[pref_M];
    int nb_count = 0;
    for (int k = 0; k < count; ++k)
      newblocks[nb_count++] = remaining[k];
    while (nb_count < (int)pref_M)
      newblocks[nb_count++] = next_prefetch++;
    for (uint32_t i = 0; i < pref_M; ++i)
      sb_blocks[buf_p_index * pref_M + i] = newblocks[i];
    sb_top[buf_p_index] = 0;
    sb_mark_mru(buf_p_index, sb_rec, pref_N);
    delete[] remaining;
    delete[] newblocks;
  }
  int sb_lru(uint32_t pref_N, bool* sb_valid, int* sb_rec)
  {
    if (pref_N == 0) return -1;
    for (size_t i = 0; i < pref_N; ++i)
        if (!sb_valid[i]) return (int)i;
    int best_p_index = 0;
    int best_rec = sb_rec[0];
    for (size_t i = 1; i < pref_N; ++i) {
        if (sb_rec[i] > best_rec) {
            best_rec = sb_rec[i];
            best_p_index = (int)i;
        }
    }
    return best_p_index;
  }
  void l2_cache(int* l2_tag_storage, int* l2_arr_valid, char* l2_arr_dirty, bool* sb_valid, int* sb_top, int* sb_rec, uint32_t* sb_blocks, uint32_t pref_M, uint32_t pref_N, cache_params& l2_counters, int l2_block_size, int l2_size, int l2_assoc, uint32_t l2_addr, char l2_rw, int* l2_arr_lru, bool is_prefetch, uint32_t* l2_arr_dirty_addr)
	{
	    auto [l2_set_bits, l2_blockoffset_bits, l2_no_of_sets] = set_tag_bits_cal(l2_block_size, l2_size, l2_assoc);
	    auto [l2_tag, l2_set] = parseaddress(l2_addr, l2_set_bits, l2_blockoffset_bits);
      if (l2_rw == 'r') {
        if (is_prefetch)
        l2_counters.prefetch_reads_from_l1++;
      }
      bool l2_search_tag = false;
      int index_tag = -1;
      for (int i = 0; i < l2_assoc; ++i) {
      if (l2_arr_valid[l2_set*l2_assoc + i] == 1 && l2_tag == l2_tag_storage[l2_set *l2_assoc + i]) {
        l2_search_tag = true;
        index_tag = i;
        break;
      }
      }

      uint32_t block_number = (l2_addr >> l2_blockoffset_bits);
      int sb_hit_p_index = -1;
      if (pref_N > 0) {
      sb_hit_p_index = sb_hit(pref_N, pref_M, sb_valid, sb_top, sb_rec,sb_blocks, block_number);
      }

      if (l2_search_tag)
      {
          lru_policy(l2_assoc, l2_set, index_tag, l2_arr_lru);
          if (l2_rw == 'r') {
          if (!is_prefetch)
            l2_counters.read_hit++;
          } else {
          l2_counters.write_hit++;
          l2_arr_dirty[l2_set * l2_assoc + index_tag] = 'D';
          }
          if (sb_hit_p_index != -1) {
            int pos = sb_pos_finder(sb_hit_p_index, block_number, pref_N, pref_M, sb_top, sb_blocks);
            if (pos >= 0) {
            sb_hit_continue(sb_hit_p_index, pos, pref_M, pref_N,
              sb_top, sb_blocks, sb_rec );
              uint32_t new_count = pos + 1;
              for (uint32_t i = 0; i < new_count; i++)
                l2_counters.prefetch_req++;
            }
          }
      }
      else
{
    if (sb_hit_p_index == -1)
    {
        if (is_prefetch)
            l2_counters.prefetch_misses_from_l1++;
        else {
            if (l2_rw == 'r') l2_counters.read_miss++;
            else l2_counters.write_miss++;
        }
    }
    else
    {
        if (!is_prefetch) {
            if (l2_rw == 'r') l2_counters.sb_read_hits++;
            else l2_counters.sb_write_hits++;
        }
    }
    for (int j = 0; j < l2_assoc; ++j) {
        if (l2_arr_lru[l2_set * l2_assoc + j] == (l2_assoc - 1)) {
            if (l2_arr_dirty[l2_set * l2_assoc + j] == 'D')
                l2_counters.writeback_to_lower_mem++;
            l2_tag_storage[l2_set * l2_assoc + j] = l2_tag;
            l2_arr_dirty[l2_set * l2_assoc + j] = (l2_rw == 'w') ? 'D' : ' ';
            l2_arr_valid[l2_set * l2_assoc + j] = 1;
            lru_policy(l2_assoc, l2_set, j, l2_arr_lru);
            break;
        }
    }
    if (sb_hit_p_index == -1) {
        if (pref_N > 0 && pref_M > 0 && !is_prefetch) {
            int sbp = sb_lru(pref_N, sb_valid, sb_rec);
            if (sbp != -1) {
                sb_miss(sbp,block_number, sb_valid, sb_top, sb_blocks, pref_M, pref_N, sb_rec);
                for (uint32_t i = 0; i < pref_M; ++i)
                    l2_counters.prefetch_req++;
            }
        }
        } else {
            int pos = sb_pos_finder(sb_hit_p_index, block_number, pref_N, pref_M, sb_top, sb_blocks);
            if (pos >= 0) {
            sb_hit_continue(sb_hit_p_index, pos, pref_M, pref_N, sb_top, sb_blocks, sb_rec);
            uint32_t new_count = pos + 1;
            for (uint32_t i = 0; i < new_count; i++)
                l2_counters.prefetch_req++;
        }
    }
  }
  }

  void l1_cache(int* l1_tag_storage, char* l1_arr_dirty, int* l1_arr_valid, uint32_t* l1_arr_dirty_addr, int* l1_arr_lru,
              int* l2_tag_storage, char* l2_arr_dirty, int* l2_arr_valid, uint32_t* l2_arr_dirty_addr,
              int* l2_arr_lru, cache_params& l1_counters, cache_params& l2_counters,
              int cache_block_size, int cache_size, int c_assoc, uint32_t l1_addr, char rw,
              int l2_size, int l2_assoc, bool* sb_valid, int* sb_top, int* sb_rec, uint32_t* sb_blocks,
              uint32_t l1_pref_N, uint32_t l1_pref_M, uint32_t l2_pref_N, uint32_t l2_pref_M)
{
    auto [l1_set_bits, l1_blockoffset_bits, l1_no_of_sets] = set_tag_bits_cal(cache_block_size, cache_size, c_assoc);
    auto [c_tag, c_set] = parseaddress(l1_addr, l1_set_bits, l1_blockoffset_bits);

    bool l1_search_tag = false;
    int index_tag = -1;
    for (int i = 0; i < c_assoc; ++i) {
        if (l1_arr_valid[c_set * c_assoc + i] == 1 && c_tag == l1_tag_storage[c_set * c_assoc + i]) {
            l1_search_tag = true;
            index_tag = i;
            break;
        }
    }

    uint32_t block_number = (l1_addr >> l1_blockoffset_bits);
    int sb_hit_p_index = -1;
    if (l1_pref_N > 0) {
        sb_hit_p_index = sb_hit(l1_pref_N, l1_pref_M, sb_valid, sb_top, sb_rec, sb_blocks, block_number);
    }

    if (l1_search_tag)
    {
        // ---- cache hit ----
        lru_policy(c_assoc, c_set, index_tag, l1_arr_lru);
        if (rw == 'r') {
            l1_counters.read_hit++;
        } else {
            l1_counters.write_hit++;
            l1_arr_dirty[c_set * c_assoc + index_tag] = 'D';
            l1_arr_dirty_addr[c_set * c_assoc + index_tag] = l1_addr;
        }

        // hit + stream buffer hit
        if (sb_hit_p_index != -1) {
            int pos = sb_pos_finder(sb_hit_p_index, block_number, l1_pref_N, l1_pref_M, sb_top, sb_blocks);
            if (pos >= 0) {
                sb_hit_continue(sb_hit_p_index, pos, l1_pref_M, l1_pref_N, sb_top, sb_blocks, sb_rec);
                uint32_t new_count = pos + 1;
                uint32_t tail_start = (sb_top[sb_hit_p_index] + l1_pref_M - new_count) % l1_pref_M;
                for (uint32_t i = 0; i < new_count; i++) {
                    uint32_t real_index = (tail_start + i) % l1_pref_M;
                    uint32_t blk = sb_blocks[sb_hit_p_index * l1_pref_M + real_index];
                    uint32_t prefetch_addr = blk << l1_blockoffset_bits;
                    l1_counters.prefetch_req++;
                    if (l2_size != 0)
                        l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                 sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                 l2_assoc, prefetch_addr, 'r', l2_arr_lru, true, l2_arr_dirty_addr);
                }
            }
        }
    }
    else
    {
        // ---- cache miss ----
        if (sb_hit_p_index == -1)
        {
            // miss + stream buffer miss
            if (rw == 'r') l1_counters.read_miss++;
            else l1_counters.write_miss++;

            for (int j = 0; j < c_assoc; ++j) {
                if (l1_arr_lru[c_set * c_assoc + j] == (c_assoc - 1)) {
                    if (l1_arr_dirty[c_set * c_assoc + j] == 'D') {
                        l1_counters.writeback_to_lower_mem++;
                        if (l2_size != 0)
                            l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                     sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                     l2_assoc, l1_arr_dirty_addr[c_set * c_assoc + j], 'w', l2_arr_lru, false, l2_arr_dirty_addr);
                    }
                    if (l2_size != 0)
                        l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                 sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                 l2_assoc, l1_addr, 'r', l2_arr_lru, false, l2_arr_dirty_addr);

                    l1_tag_storage[c_set * c_assoc + j] = c_tag;
                    lru_policy(c_assoc, c_set, j, l1_arr_lru);
                    l1_arr_valid[c_set * c_assoc + j] = 1;
                    if (rw == 'w') {
                        l1_arr_dirty[c_set * c_assoc + j] = 'D';
                        l1_arr_dirty_addr[c_set * c_assoc + j] = l1_addr;
                    } else {
                        l1_arr_dirty[c_set * c_assoc + j] = ' ';
                        l1_arr_dirty_addr[c_set * c_assoc + j] = 0;
                    }
                    break;
                }
            }

            if (l1_pref_N > 0 && l1_pref_M > 0) {
                int sbp = sb_lru(l1_pref_N, sb_valid, sb_rec);
                if (sbp != -1) {
                    sb_miss(sbp, block_number, sb_valid, sb_top, sb_blocks, l1_pref_M, l1_pref_N, sb_rec);
                    for (uint32_t i = 0; i < l1_pref_M; ++i) {
                        uint32_t blk = sb_blocks[sbp * l1_pref_M + i];
                        uint32_t prefetch_addr = blk << l1_blockoffset_bits;
                        l1_counters.prefetch_req++;
                        if (l2_size != 0)
                            l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                     sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                     l2_assoc, prefetch_addr, 'r', l2_arr_lru, true, l2_arr_dirty_addr);
                    }
                }
            }
        }
        else
        {
            // miss + stream buffer hit
            if (rw == 'r') l1_counters.sb_read_hits++;
            else l1_counters.sb_write_hits++;

            for (int j = 0; j < c_assoc; ++j) {
                if (l1_arr_lru[c_set * c_assoc + j] == (c_assoc - 1)) {
                    if (l1_arr_dirty[c_set * c_assoc + j] == 'D') {
                        l1_counters.writeback_to_lower_mem++;
                        if (l2_size != 0)
                            l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                     sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                     l2_assoc, l1_arr_dirty_addr[c_set * c_assoc + j], 'w', l2_arr_lru, false, l2_arr_dirty_addr);
                    }
                    if (rw == 'w' && l2_size != 0)
                        l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                 sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                 l2_assoc, l1_addr, 'r', l2_arr_lru, false, l2_arr_dirty_addr);

                    l1_tag_storage[c_set * c_assoc + j] = c_tag;
                    lru_policy(c_assoc, c_set, j, l1_arr_lru);
                    l1_arr_valid[c_set * c_assoc + j] = 1;
                    if (rw == 'w') {
                        l1_arr_dirty[c_set * c_assoc + j] = 'D';
                        l1_arr_dirty_addr[c_set * c_assoc + j] = l1_addr;
                    } else {
                        l1_arr_dirty[c_set * c_assoc + j] = ' ';
                        l1_arr_dirty_addr[c_set * c_assoc + j] = 0;
                    }
                    break;
                }
            }

            int pos = sb_pos_finder(sb_hit_p_index, block_number, l1_pref_N, l1_pref_M, sb_top, sb_blocks);
            if (pos >= 0) {
                sb_hit_continue(sb_hit_p_index, pos, l1_pref_M, l1_pref_N, sb_top, sb_blocks, sb_rec);
                uint32_t new_count = pos + 1;
                uint32_t tail_start = (sb_top[sb_hit_p_index] + l1_pref_M - new_count) % l1_pref_M;
                for (uint32_t i = 0; i < new_count; i++) {
                    uint32_t real_index = (tail_start + i) % l1_pref_M;
                    uint32_t blk = sb_blocks[sb_hit_p_index * l1_pref_M + real_index];
                    uint32_t prefetch_addr = blk << l1_blockoffset_bits;
                    l1_counters.prefetch_req++;
                    if (l2_size != 0)
                        l2_cache(l2_tag_storage, l2_arr_valid, l2_arr_dirty, sb_valid, sb_top, sb_rec,
                                 sb_blocks, l2_pref_M, l2_pref_N, l2_counters, cache_block_size, l2_size,
                                 l2_assoc, prefetch_addr, 'r', l2_arr_lru, true, l2_arr_dirty_addr);
                }
            }
        }
    }
}


#endif
