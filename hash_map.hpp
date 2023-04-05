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

    int get_target_rank(const uint64_t &key);
    int get_target_slot(const uint64_t &key);
};

HashMap::HashMap(size_t size) {
    my_size = size;
    my_global_size = my_size * upcxx::rank_n();
    // data_global = (int*) upc_alloc(size, sizeof(kmer_pair));
    // used_global = (int*) upc_alloc(size, sizeof(int));
    data_global = upcxx::new_array<kmer_pair>(size);
    used_global = upcxx::new_array<int>(size);
    int nprocs = upcxx::rank_n();
    shared_data = std::vector<upcxx::global_ptr<kmer_pair>> (nprocs);
    shared_used = std::vector<upcxx::global_ptr<int>> (nprocs);
    used = used_global.local();
    data = data_global.local();

    memset(used, 0, size);

    for (int i = 0; i < nprocs; i++){
        shared_data[i] = upcxx::broadcast(data_global, i).wait();
        shared_used[i] = upcxx::broadcast(used_global, i).wait();
    }

    // data.resize(size);
    // used.resize(size, 0);
}
int HashMap::get_target_rank(const uint64_t &key) {
    return (key) / size();
}
int HashMap::get_target_slot(const uint64_t &key) {
    return (key) % size();
}
bool HashMap::insert(const kmer_pair& kmer) {
    //get rank
    // std::cout << "in insert" << shared_used[0] << "\n";
    // std::cout << upcxx::rank_me() << " in insert used global " << used_global << " mysize " << my_size << "\n";
    uint64_t hash = kmer.hash();

    //linear probing
    uint64_t probe = 0;
    bool success = false;
    do {
        std::cout << "at start \n";
        uint64_t slot = (hash + probe++) % global_size();
        success = request_slot(slot);
        std::cout << probe << "\n";
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
        uint64_t slot = (hash + probe++) % global_size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < global_size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { 
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    upcxx::atomic_domain<int> ad_int({upcxx::atomic_op::load});
    int used_val = ad_int.load(shared_used[target_rank] + target_slot, std::memory_order_relaxed).wait();
    ad_int.destroy();
    return used_val != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    std::cout << slot << " in write slot my rank: " << upcxx::rank_me() << " rank target: " << target_rank << "\n";
    // rput(kmer, shared_data[target_rank] + target_slot).wait();
    std::cout << slot << " finished write slot my rank: " << upcxx::rank_me() << " rank target: " << target_rank << "\n";
    // data[slot] = kmer; 
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    return rget(shared_data[target_rank] + target_slot).wait();
}

bool HashMap::request_slot(uint64_t slot) {
    int target_rank = get_target_rank(slot);
    int target_slot = get_target_slot(slot);
    std::cout << size() << " slot: " << slot << " target rank " << target_rank << " target_slot " << target_slot << "\n";
    upcxx::atomic_domain<int> ad_int({upcxx::atomic_op::load,upcxx::atomic_op::fetch_add});
    int prev = ad_int.fetch_add(shared_used[target_rank] + target_slot, 1, std::memory_order_relaxed).wait();
    // cout << "saw a " << prev << "in "
    // upcxx::global_ptr<int> hits = upcxx::broadcast(upcxx::new_<int>(0), 0).wait();
    // shared_used[target_rank] + target_slot
    ad_int.destroy();
    return (prev == 0);
    // return 1;
}

size_t HashMap::size() const noexcept { return my_size; }
size_t HashMap::global_size() const noexcept { return my_global_size; }
