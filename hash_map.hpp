#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    // shared std::vector<kmer_pair> data;
    // shared std::vector<int> used;

    upcxx::global_ptr<kmer_pair> data_global;
    upcxx::global_ptr<int> used_global;
    int* used;
    kmer_pair* data;
    //TODO: MAKE IT LOCAL FOR OPTIMIZATION LATER

    std::vector<upcxx::global_ptr<kmer_pair>> shared_data;
    std::vector<upcxx::global_ptr<int>> shared_used;
    size_t my_size;
    size_t my_global_size;
    size_t size() const noexcept;
    size_t global_size() const noexcept;

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
    my_size = size;
    my_global_size = my_size * upcxx::rank_n();
    upcxx::global_ptr<kmer_pair> data_global = upcxx::new_array<kmer_pair>(size);
    upcxx::global_ptr<int> used_global = upcxx::new_array<int>(size);
    int nprocs = upcxx::rank_n();
    std::vector<upcxx::global_ptr<kmer_pair>> shared_data = std::vector<upcxx::global_ptr<kmer_pair>> (nprocs);
    std::vector<upcxx::global_ptr<int>> shared_used = std::vector<upcxx::global_ptr<int>> (nprocs);
    for (int i = 0; i < nprocs; i++){
        shared_data[i] = upcxx::broadcast(data_global, i).wait();
        shared_used[i] = upcxx::broadcast(used_global, i).wait();
    }

    used = used_global.local();
    data = data_global.local();
    // data.resize(size);
    // used.resize(size, 0);
}
int get_target_rank(const uint64_t &key) {
    return (key) / upcxx::rank_n();
}
int get_target_slot(const uint64_t &key) {
    return (key) % upcxx::rank_n();
}
bool HashMap::insert(const kmer_pair& kmer) {
    //get rank
    uint64_t hash = kmer.hash();

    //linear probing
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % global_size();

        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < global_size());
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { return used[slot] != 0; }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    rput(kmer, shared_data[target_rank] + target_slot).wait();
    // data[slot] = kmer; 
}

kmer_pair HashMap::read_slot(uint64_t slot) { return data[slot]; }

bool HashMap::request_slot(uint64_t slot) {
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    int used = rget(shared_used[target_rank] + target_slot).wait();
    upcxx::atomic_domain<int> ad_i64({upcxx::atomic_op::load, upcxx::atomic_op::fetch_add});
    int prev = ad_i64.fetch_add(shared_used[target_rank] + target_slot, 1, std::memory_order_relaxed).wait();

    return (prev != 0);
    
    // if (used != 0) {
    //     return false;
    // } else {
    //     rput(, 1).wait();
    //     return true;
    // }
}

size_t HashMap::size() const noexcept { return my_size; }
size_t HashMap::global_size() const noexcept { return my_global_size; }
