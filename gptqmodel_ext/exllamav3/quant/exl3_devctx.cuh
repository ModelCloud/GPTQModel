#pragma once

#include <tuple>
#include <mutex>

// Workspace size
#define WORKSPACE_SIZE (4*1024*1024)

// Treat hopper and blackwell as same arch for now
#define MAX_DEVICES 16
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  4

// Singleton for the read-only per-device context: device attributes that never
// change while the process runs. Writable kernel scratch (the split-K lock
// array, the hgemm workspace) is *not* stored here: it belongs to one call, and
// a per-device copy was shared by every stream (issue #4619).
class DevCtx
{
private:
    int num_sms[MAX_DEVICES] = {};
    int cc[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance();
    int get_num_sms(int device);
    int get_cc(int device);

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);

void prepare_ctx(int device);
