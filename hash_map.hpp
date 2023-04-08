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
    upcxx::atomic_domain<int> ad_int;

    upcxx::future<> fut_all;
    int batch_size;
    int curr_count;
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
    void write_slot(uint64_t target_rank, uint64_t target_slot, const kmer_pair& kmer);

    kmer_pair read_slot(uint64_t target_rank, uint64_t target_slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t target_rank, uint64_t target_slot);
    bool slot_used(uint64_t target_rank, uint64_t target_slot);

    int get_target_rank(const uint64_t &key);
    int get_target_slot(const uint64_t &key);

    void flush_writes(void);
};

HashMap::HashMap(size_t size) {
    my_size = size;
    my_global_size = my_size * upcxx::rank_n();

    ad_int = upcxx::atomic_domain<int>({upcxx::atomic_op::load, upcxx::atomic_op::fetch_add});

    data_global = upcxx::new_array<kmer_pair>(size);
    used_global = upcxx::new_array<int>(size);
    int nprocs = upcxx::rank_n();
    shared_data = std::vector<upcxx::global_ptr<kmer_pair>> (nprocs);
    shared_used = std::vector<upcxx::global_ptr<int>> (nprocs);
    used = used_global.local();
    data = data_global.local();

    memset(used, 0, size);

    fut_all = upcxx::make_future();
    batch_size = 0.01*my_size;
    curr_count = batch_size;

    for (int i = 0; i < nprocs; i++){
        shared_data[i] = upcxx::broadcast(data_global, i).wait();
        shared_used[i] = upcxx::broadcast(used_global, i).wait();
    }
}
int HashMap::get_target_rank(const uint64_t &key) {
    return (key) / size();
}
int HashMap::get_target_slot(const uint64_t &key) {
    return (key) % size();
}
bool HashMap::insert(const kmer_pair& kmer) {
    //get rank
    uint64_t hash = kmer.hash();

    //linear probing
    uint64_t probe = 0;
    bool success = false;
    uint64_t slot = (hash) % global_size();
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    do {
        success = request_slot(target_rank, target_slot);

        if (success) {
            write_slot(target_rank, target_slot, kmer);
            break;
        }
        probe++;
        target_slot++;
        if (target_slot == size()) {
            target_slot = 0;
            target_rank++;
            if (target_rank == upcxx::rank_n())
                target_rank = 0;
        }
    } while (!success && probe < global_size());
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    
    uint64_t slot = (hash) % global_size();
    uint64_t target_rank = get_target_rank(slot);
    uint64_t target_slot = get_target_slot(slot);
    do {
        if (slot_used(target_rank, target_slot)) {
            val_kmer = read_slot(target_rank, target_slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
                break;
            }
        }
        probe++;
        target_slot++;
        if (target_slot == size()) {
            target_slot = 0;
            target_rank++;
            if (target_rank == upcxx::rank_n())
                target_rank = 0;
        }
    } while (!success && probe < global_size());
    return success;
}

bool HashMap::slot_used(uint64_t target_rank, uint64_t target_slot) { 
    int used_val = ad_int.load(shared_used[target_rank] + target_slot, std::memory_order_relaxed).wait();
    return used_val != 0;
}

void HashMap::write_slot(uint64_t target_rank, uint64_t target_slot, const kmer_pair& kmer) { 
    upcxx::future<> fut = rput(kmer, shared_data[target_rank] + target_slot);
    fut_all = upcxx::when_all(fut_all, fut);
    curr_count -= 1;
    if (batch_size == 0) {
        fut_all.wait();
        curr_count = batch_size;
    }
}

kmer_pair HashMap::read_slot(uint64_t target_rank, uint64_t target_slot) { 
    return rget(shared_data[target_rank] + target_slot).wait();
}

bool HashMap::request_slot(uint64_t target_rank, uint64_t target_slot) {
    int prev = ad_int.fetch_add(shared_used[target_rank] + target_slot, 1, std::memory_order_relaxed).wait();
    return (prev == 0);
}

void HashMap::flush_writes(void) {
    fut_all.wait();
}
size_t HashMap::size() const noexcept { return my_size; }
size_t HashMap::global_size() const noexcept { return my_global_size; }
