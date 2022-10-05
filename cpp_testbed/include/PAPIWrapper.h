//
// Created by kazem on 2020-05-26.
//

#ifndef FUSION_PAPIWRAPPER_H
#define FUSION_PAPIWRAPPER_H

#include <papi.h>
#include <iomanip>
#include <utility>
#include <vector>

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }
#define PAPI_CHECK(papi_cmd) do { int rc; if((rc = papi_cmd) != PAPI_OK) ERROR_RETURN(rc); } while(0)

namespace cpp_testbed {
    struct PapiEvent {
        int event_code;
        std::string event_name;
        std::string alt_name;

        union {
            int64_t i = 0;
            double   d;
        } result;

        enum {
            DOUBLE,
            INTEGER
        } storage = INTEGER;
    };

    std::string convert_code_to_string(int code){
        char EventCodeStr[PAPI_MAX_STR_LEN];
        PAPI_CHECK(PAPI_event_code_to_name(code, EventCodeStr));
        std::string ret(EventCodeStr);
        return ret;
    }

    int event_name_to_code(std::string name) {
        int event_code;
        int retval = 0;
        retval = PAPI_event_name_to_code(name.c_str(), &event_code);
        if (retval != PAPI_OK) {
            std::cerr << "Failed to convert event " << name << " to code" << std::endl;
            ERROR_RETURN(retval);
        }

        //std::cout << name << " " << event_code << std::endl;
        return event_code;
    }

    void populate_papi_event(std::vector<PapiEvent>& events, std::string name, std::string alt_name = "") {
        PapiEvent event;
        event.event_name = name;
        event.alt_name = alt_name.empty() ? name : alt_name;
        event.event_code = event_name_to_code(name);
        events.push_back(event);
    }

    void populate_papi_event(std::vector<PapiEvent>& events, int event_code, std::string alt_name = "") {
        PapiEvent event;
        std::string name = convert_code_to_string(event_code);
        event.event_name = name;
        event.alt_name = alt_name.empty() ? name : alt_name;
        event.event_code = event_code;
        events.push_back(event);
    }

    uint64_t get_value_based_on_alt_name(std::vector<struct PapiEvent>& events, std::string alt_name) {
        for (const auto& event : events) {
            if (event.alt_name == alt_name) {
                assert(event.storage == PapiEvent::INTEGER);
                return event.result.i;
            }
        }

        std::cerr << "Failed to find counter result " << alt_name << std::endl;
        exit(-1);
        return 0;
    }


#ifdef RASPBERRY_PI
    // Based off of: https://github.com/avr-aics-riken/PMlib/blob/master/src/PerfCpuType.cpp
    std::vector<
        std::pair<
                std::function<std::vector<struct PapiEvent>(void)>,
                std::function<std::vector<struct PapiEvent>(std::vector<struct PapiEvent>)
            >
        >
    > EVENT_CAPTURE_SETS = {
        {   // Calculate Data cache
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "L1D_READ_ACCESS",    "LOADS");
                populate_papi_event(events, "L1D_WRITE_ACCESS",   "STORES");
                populate_papi_event(events, "L1D_CACHE_ACCESS",   "L1_ACCESS_TOTAL");
                populate_papi_event(events, "L1D_CACHE_REFILL",   "L1_MISS_TOTAL");
                return events;
            },
            [](std::vector<struct PapiEvent> events) {

                int64_t ACCESSES = get_value_based_on_alt_name(events, "L1_ACCESS_TOTAL");
                int64_t MISSES   = get_value_based_on_alt_name(events, "L1_MISS_TOTAL");
                int64_t STORES   = get_value_based_on_alt_name(events, "STORES");
                int64_t LOADS   = get_value_based_on_alt_name(events, "LOADS");

                double L1_HIT_RATE = (double) (ACCESSES - MISSES) / (double) ACCESSES;
                std::cout << "L1HIT: " << std::setprecision(2) << L1_HIT_RATE
                          << " AC: " << ACCESSES
                          << " ST: " << STORES
                          << " LD: " << LOADS
                          << " ";
                events.push_back({-1, "L1_RATIO", "L1_RATIO", {.d = L1_HIT_RATE}, PapiEvent::DOUBLE});

                return events;
            }
        },
        {   // Calculate Instruction cache + some bonus data cache metrics
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "L1D_READ_REFILL",    "L1_READ_MISS");
                populate_papi_event(events, "L1D_WRITE_REFILL",   "L1_STORE_MISS");
                populate_papi_event(events, "L1I_CACHE_ACCESS",   "L1I_ACCESS_TOTAL");
                populate_papi_event(events, "L1I_CACHE_REFILL",   "L1I_MISS_TOTAL");
                return events;
            },
            [](std::vector<struct PapiEvent> events) {

                int64_t ACCESSES = get_value_based_on_alt_name(events, "L1I_ACCESS_TOTAL");
                int64_t MISSES   = get_value_based_on_alt_name(events, "L1I_MISS_TOTAL");

                double L1_MISS_RATE = (double) (ACCESSES - MISSES) / (double) ACCESSES;
                events.push_back({-1, "L1I_RATIO", "L1I_RATIO", {.d = L1_MISS_RATE}, PapiEvent::DOUBLE});

                return events;
            }
        },
        {   // Calculate TLB
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "L1D_TLB_REFILL", "L1_TLB");
                populate_papi_event(events, "L1I_TLB_REFILL", "L1I_TLB");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },

        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "BRANCH_PRED",      "BRANCH_PRED");
                populate_papi_event(events, "BRANCH_MISPRED",   "BRANCH_MISPRED");
                populate_papi_event(events, "INST_RETIRED",     "INST_RETIRED");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },

        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "UNALIGNED_ACCESS",      "UNALIGNED_ACCESS");
                populate_papi_event(events, "DATA_MEM_ACCESS",       "DATA_MEM_ACCESS");
                populate_papi_event(events, "DATA_MEM_READ_ACCESS",  "DATA_MEM_READ_ACCESS");
                populate_papi_event(events, "DATA_MEM_WRITE_ACCESS", "DATA_MEM_WRITE_ACCESS");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },

        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "BUS_NORMAL_ACCESS",     "BUS_NORMAL_ACCESS");
                populate_papi_event(events, "BUS_READ_ACCESS",       "BUS_READ_ACCESS");
                populate_papi_event(events, "BUS_WRITE_ACCESS",      "BUS_WRITE_ACCESS");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },

        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                populate_papi_event(events, "INST_SPEC_EXEC_LOAD",   "INST_SPEC_EXEC_LOAD");
                populate_papi_event(events, "INST_SPEC_EXEC_STORE",  "INST_SPEC_EXEC_STORE");
                populate_papi_event(events, "INST_SPEC_EXEC_SIMD",   "INST_SPEC_EXEC_SIMD");
                populate_papi_event(events, "INST_SPEC_EXEC_VFP",    "INST_SPEC_EXEC_VFP");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        }
    };
#else
    // Based off of: https://github.com/avr-aics-riken/PMlib/blob/master/src/PerfCpuType.cpp
    std::vector<
        std::pair<
                std::function<std::vector<struct PapiEvent>(void)>,
                std::function<std::vector<struct PapiEvent>(std::vector<struct PapiEvent>)
            >
        >
    > EVENT_CAPTURE_SETS = {
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "MEM_UOPS_RETIRED:ALL_LOADS",    "UOPS_LOADS");
                populate_papi_event(events, "MEM_UOPS_RETIRED:ALL_STORES",   "UOPS_STORES");
                populate_papi_event(events, "MEM_UOPS_RETIRED:SPLIT_LOADS",  "UOPS_SPLIT_LOADS");
                populate_papi_event(events, "MEM_UOPS_RETIRED:SPLIT_STORES", "UOPS_SPLIT_STORES");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "UOPS_DISPATCHED:PORT_0", "PORT_0");
                populate_papi_event(events, "UOPS_DISPATCHED:PORT_1", "PORT_1");
                populate_papi_event(events, "UOPS_DISPATCHED:PORT_5", "PORT_5");
                populate_papi_event(events, "UNHALTED_CORE_CYCLES", "CYCLES_1");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "UOPS_DISPATCHED:PORT_2", "PORT_2");
                populate_papi_event(events, "UOPS_DISPATCHED:PORT_3", "PORT_3");
                populate_papi_event(events, "UOPS_DISPATCHED:PORT_4", "PORT_4");
                populate_papi_event(events, "UNHALTED_CORE_CYCLES", "CYCLES_2");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "UOPS_DISPATCHED:PORT_6",   "PORT_6");
                populate_papi_event(events, "UOPS_DISPATCHED:PORT_7",   "PORT_7");
                populate_papi_event(events, "UOPS_ISSUED:SLOW_LEA",     "SLOW_LEA");
                populate_papi_event(events, "UNHALTED_CORE_CYCLES",     "CYCLES_3");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, PAPI_TOT_CYC, "TOT_CYC");
                populate_papi_event(events, PAPI_TOT_INS, "TOT_INS");
                populate_papi_event(events, PAPI_LD_INS, "LOAD_INS");
                populate_papi_event(events, PAPI_SR_INS, "STORE_INS");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Calculate Flops Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "FP_ARITH:SCALAR_SINGLE",      "SP_SINGLE");
                populate_papi_event(events, "FP_ARITH:128B_PACKED_SINGLE", "SP_SSE");
                populate_papi_event(events, "FP_ARITH:256B_PACKED_SINGLE", "SP_AVX");
                populate_papi_event(events, "FP_ARITH:512B_PACKED_SINGLE", "SP_AVXW");

                populate_papi_event(events, PAPI_SP_OPS, "SP_OPS");

                // TODO: Double support

                // populate_papi_event(events[idx++], "DP_OPS", "SP_AVXW");

                return events;
            },
            [](std::vector<struct PapiEvent> events) {
                int64_t SP_SINGLE = get_value_based_on_alt_name(events, "SP_SINGLE");
                int64_t SP_SSE    = get_value_based_on_alt_name(events, "SP_SSE");
                int64_t SP_AVX    = get_value_based_on_alt_name(events, "SP_AVX");
                int64_t SP_AVXW   = get_value_based_on_alt_name(events, "SP_AVXW");

                int64_t SP_VECTOR = 4 * SP_SSE + 8 * SP_AVX + 16 * SP_AVXW;
                int64_t SP_FLOPS_TOTAL = SP_SINGLE + SP_VECTOR;
                events.push_back({ -1, "SP_FLOPS_TOTAL", "SP_FLOPS_TOTAL", SP_FLOPS_TOTAL });
                events.push_back({ -1, "SP_VECTOR",      "SP_VECTOR",      SP_VECTOR });

                return events;
            }
        },
        {   // Calculate L1/L2 Cache Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "MEM_LOAD_UOPS_RETIRED:L1_HIT",  "L1_HIT");
                populate_papi_event(events, "MEM_LOAD_UOPS_RETIRED:HIT_LFB", "LFB_HIT");

                populate_papi_event(events, PAPI_L1_TCM, "L1_TCM");


                return events;
            },
            [](std::vector<struct PapiEvent> events) {
                int64_t L1_HIT    = get_value_based_on_alt_name(events, "L1_HIT");
                int64_t LFB_HIT   = get_value_based_on_alt_name(events, "LFB_HIT");
                int64_t L1_MISS   = get_value_based_on_alt_name(events, "L1_TCM");

                int64_t L1_DCA = L1_HIT + LFB_HIT + L1_MISS;


                double L1_RATIO = (double(L1_HIT) + double(LFB_HIT)) / \
                              (double(L1_HIT) + double(LFB_HIT) + double(L1_MISS));

                events.push_back({-1, "L1_DCA", "L1_DCA", {.i = L1_DCA}, PapiEvent::INTEGER});
                events.push_back({-1, "L1_RATIO", "L1_RATIO", {.d = L1_RATIO}, PapiEvent::DOUBLE});

                return events;
            }
        },
        {   // Calculate L3 Cache Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, PAPI_L1_TCM, "L1_TCM_2");
                populate_papi_event(events, PAPI_L2_TCM, "L2_TCM");

                return events;
            },
            [](std::vector<struct PapiEvent> events) {
                int64_t L1_MISS   = get_value_based_on_alt_name(events, "L1_TCM_2");
                int64_t L2_MISS   = get_value_based_on_alt_name(events, "L2_TCM");

                int64_t L2_HIT = L1_MISS - L2_MISS;

                double L2_RATIO = double(L2_HIT) / double(L1_MISS);
                events.push_back({-1, "L2_RATIO", "L2_RATIO", {.d = L2_RATIO}, PapiEvent::DOUBLE});


                return events;
            }
        },
        {   // Calculate L3 Cache Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;

                populate_papi_event(events, "OFFCORE_RESPONSE_0:ANY_DATA:L3_HIT",  "L3_HIT");
                populate_papi_event(events, "OFFCORE_RESPONSE_0:ANY_DATA:L3_MISS", "L3_MISS");

                populate_papi_event(events, "L2_RQSTS:DEMAND_DATA_RD_HIT", "L2_RD_HIT");
                populate_papi_event(events, "L2_RQSTS:PF_HIT",             "L2_PF_HIT");

                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Branch Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;
                populate_papi_event(events, PAPI_BR_UCN, "BR_UCN");
                populate_papi_event(events, PAPI_BR_CN,  "BR_CN");
                populate_papi_event(events, PAPI_BR_MSP, "BR_MSP");
                populate_papi_event(events, PAPI_BR_PRC, "BR_PRC");

                return events;
            },
            [](std::vector<struct PapiEvent> events) {
                int64_t BR_CN  = get_value_based_on_alt_name(events, "BR_CN");
                int64_t BR_MSP = get_value_based_on_alt_name(events, "BR_MSP");

                double BR_MSP_RATIO = double(BR_MSP) / double(BR_CN);
                events.push_back({-1, "BR_MSP_RATIO", "BR_MSP_RATIO", {.d = BR_MSP_RATIO}, PapiEvent::DOUBLE});

                return events;
            }
        },
        {   // Stall Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;
                populate_papi_event(events, "CYCLE_ACTIVITY:STALLS_TOTAL",    "STALLS_TOTAL");
                populate_papi_event(events, "CYCLE_ACTIVITY:CYCLES_L1D_MISS", "CYCLES_L1D_MISS");
                populate_papi_event(events, "CYCLE_ACTIVITY:CYCLES_L2_MISS",  "CYCLES_L2_MISS");
                populate_papi_event(events, "CYCLE_ACTIVITY:STALLS_L3_MISS",  "STALLS_L3_MISS");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // Stall Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;
                populate_papi_event(events, "CYCLE_ACTIVITY:STALLS_TOTAL",    "STALLS_TOTAL");
                populate_papi_event(events, "CYCLE_ACTIVITY:CYCLES_MEM_ANY",  "CYCLES_MEM_ANY");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // TLB Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;
                populate_papi_event(events, "DTLB_LOAD_MISSES:MISS_CAUSES_A_WALK",  "DTLB_LOAD_MISS");
                populate_papi_event(events, "DTLB_LOAD_MISSES:WALK_DURATION",       "DTLB_LOAD_WALK_DURATION");
                populate_papi_event(events, "DTLB_STORE_MISSES:MISS_CAUSES_A_WALK", "DTLB_STORE_MISS");
                populate_papi_event(events, "DTLB_STORE_MISSES:WALK_DURATION",      "DTLB_STORE_WALK_DURATION");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        },
        {   // TLB Metrics
            []() {
                std::vector<struct PapiEvent> events;
                int idx = 0;
                populate_papi_event(events, "ITLB_MISSES:MISS_CAUSES_A_WALK", "ITLB_MISS");
                populate_papi_event(events, "ITLB_MISSES:WALK_DURATION",      "ITLB_WALK_DURATION");
                return events;
            },
            [](std::vector<struct PapiEvent> events) { return events; }
        }
    };
#endif

    class PAPIWrapper {
        int event_set_ = PAPI_NULL; // PAPI
        int curr_capture_set_ = 0;
        std::vector<struct PapiEvent> curr_event_list_;

        std::string convert_code_to_string(int code){
            char EventCodeStr[PAPI_MAX_STR_LEN];
            PAPI_event_code_to_name(code,EventCodeStr);
            std::string ret(EventCodeStr);
            return ret;
        }

    public:

        std::vector<struct PapiEvent> get_results() { return curr_event_list_; }

        static inline std::vector<int> get_available_counter_codes() {
            std::vector<int> cc;
            int retval;
            if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) ERROR_RETURN(retval);

            PAPI_event_info_t info;
            int mask = PAPI_PRESET_ENUM_AVAIL;
            int start = 0 | PAPI_PRESET_MASK;
            int i = start;

            do {
                retval = PAPI_get_event_info(i, &info);

                if (retval == PAPI_OK) {
                    cc.push_back(info.event_code);
                }
                retval = PAPI_enum_event(&i, mask);
            }  while (retval == PAPI_OK);

            PAPI_shutdown();

            return cc;
        }

        static bool compatible_events(const std::vector<int>& event_list) {
            int event_set_ = PAPI_NULL;
            int retval;

            if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) ERROR_RETURN(retval);
            PAPI_CHECK(PAPI_create_eventset(&event_set_));

            int add_events_status = PAPI_add_events(event_set_,
                                                    const_cast<int*>(event_list.data()),
                                                    event_list.size());
            bool compatible = (add_events_status == PAPI_OK);

            PAPI_CHECK(PAPI_cleanup_eventset(event_set_));
            PAPI_CHECK(PAPI_destroy_eventset(&event_set_));

            PAPI_shutdown();

            return compatible;
        }

        PAPIWrapper() {
            event_set_ = 0;
        }

        int begin_profiling(int capture_set){
            event_set_ = PAPI_NULL;

            if (capture_set >= EVENT_CAPTURE_SETS.size()) {
                std::cerr << "Invalid capture set" << std::endl;
                exit(-1);
            }

            int retval;
            if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) ERROR_RETURN(retval);

            curr_capture_set_ = capture_set;
            curr_event_list_ = EVENT_CAPTURE_SETS[capture_set].first();
            std::vector<int> event_codes(curr_event_list_.size());

            for (int i = 0; i < curr_event_list_.size(); i++) {
                event_codes[i] = curr_event_list_[i].event_code;
            }

            PAPI_CHECK(PAPI_create_eventset(&event_set_));

            if ((retval = PAPI_add_events(event_set_, event_codes.data(), event_codes.size())) != PAPI_OK) {
                std::cerr << "Failed to create event set: ";
                for (const auto& event : curr_event_list_) std::cerr << event.event_name << " ";
                std::cerr << std::endl;
                ERROR_RETURN(retval);
            }

            PAPI_CHECK(PAPI_reset(event_set_));
            PAPI_CHECK(PAPI_start(event_set_));

            return 0;
        }

        bool finish_profiling(){
            int retval;
            bool success = true;
            std::vector<long long int> event_counters(curr_event_list_.size());
            PAPI_CHECK(PAPI_stop(event_set_, event_counters.data()));

            // Sometimes PAPI will return an abnormally large value on niagara
            //   the root cause of this is still unknown so for now just flag
            //   the profiling as failed and re-profile the whole set
            for (int j = 0; j < event_counters.size(); ++j) {
                if (event_counters[j] > 1e12) {
                    success = false;
                    std::cerr << "[PAPIWrapper] Abnormally high counter " << curr_event_list_[j].event_name;
                    std::cerr << " " << event_counters[j] << std::endl;
                }
            }

            if (success) {
                for (int j = 0; j < event_counters.size(); ++j) {
                    curr_event_list_[j].result.i = event_counters[j];
                }

                // Add derived events
                curr_event_list_ = EVENT_CAPTURE_SETS[curr_capture_set_].second(curr_event_list_);
            }

            if((retval = PAPI_cleanup_eventset(event_set_)) != PAPI_OK) std::cerr << "PAPI error cleaning-up event set " << retval << "\n";
            if((retval = PAPI_destroy_eventset(&event_set_)) != PAPI_OK) std::cerr << "PAPI error destroying event set " << retval << "\n";

            PAPI_shutdown();
            return success;
        }
    };
}

#endif