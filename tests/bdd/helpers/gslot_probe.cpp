#include "examples/stage-runner/gslot_client.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

static void print_state(const char * label, bool permitted, const gslot::gate & gate) {
    std::printf(
        "%s active=%d permitted=%d granted=%lu blocked=%lu faults=%lu burst=%d\n",
        label,
        gate.active() ? 1 : 0,
        permitted ? 1 : 0,
        gate.granted(),
        gate.blocked(),
        gate.faults(),
        gate.burst() ? 1 : 0);
    std::fflush(stdout);
}

static bool wait_for_lease(gslot::gate & gate) {
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (gate.open(gslot::now_ms_monotonic())) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "once";
    gslot::gate gate;

    if (std::strcmp(mode, "once") == 0) {
        const bool permitted = gate.open(gslot::now_ms_monotonic());
        print_state("ONCE", permitted, gate);
        return 0;
    }

    if (std::strcmp(mode, "hold") == 0) {
        const bool permitted = wait_for_lease(gate);
        print_state("HOLDING", permitted, gate);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        return permitted ? 0 : 2;
    }

    if (std::strcmp(mode, "cycle") == 0) {
        bool permitted = wait_for_lease(gate);
        print_state("FIRST", permitted, gate);
        if (!permitted) {
            return 2;
        }
        gate.yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(2100));
        permitted = wait_for_lease(gate);
        print_state("SECOND", permitted, gate);
        gate.yield();
        std::puts("READY");
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        return permitted ? 0 : 2;
    }

    if (std::strcmp(mode, "quantum") == 0) {
        bool permitted = wait_for_lease(gate);
        print_state("FIRST", permitted, gate);
        permitted = gate.open(gslot::now_ms_monotonic());
        print_state("WITHIN", permitted, gate);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        permitted = wait_for_lease(gate);
        print_state("AFTER", permitted, gate);
        gate.yield();
        return permitted ? 0 : 2;
    }

    if (std::strcmp(mode, "burst") == 0) {
        bool permitted = wait_for_lease(gate);
        print_state("ACQUIRED", permitted, gate);
        gate.handoff();
        permitted = wait_for_lease(gate);
        print_state("REACQUIRED", permitted, gate);
        gate.handoff();
        return permitted ? 0 : 2;
    }

    std::fprintf(stderr, "usage: gslot_probe once|hold|cycle|quantum|burst\n");
    return 64;
}
