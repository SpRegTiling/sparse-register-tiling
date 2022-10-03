//
// Created by kazem on 2020-05-27.
//

#ifndef FUSION_PROFILER_H
#define FUSION_PROFILER_H

//#include "FusionDemo.h"
#include "PAPIWrapper.h"
#include "csv_log_io.h"
#include <cmath>

namespace cpp_testbed {
    template<typename Functor>
    class Profiler {
        std::string name_;
        std::vector<PapiEvent> results_;

        Functor *evaluate = nullptr;

    public:
        Profiler(Functor *evaluate) : evaluate(evaluate), name_(""){}


        void profile() {
            PAPIWrapper pw;

            for (int capture_group = 0; capture_group < EVENT_CAPTURE_SETS.size(); capture_group++) {
                bool profiling_succeeded = false;
                do {
                    pw.begin_profiling(capture_group);
                    (*evaluate)();
                    profiling_succeeded = pw.finish_profiling();
                } while (!profiling_succeeded);

                auto results = pw.get_results();
                for (const auto& result : results) results_.push_back(result);
            }
        }

        void log_counters(csv_row_t& csv_row){
            for (auto & result : results_) {

                switch(result.storage) {
                    case PapiEvent::INTEGER: csv_row_insert(csv_row, result.alt_name, (int) result.result.i); break;
                    case PapiEvent::DOUBLE:  csv_row_insert(csv_row, result.alt_name, (int) result.result.d); break;
                }
            }
        }
    };
}
#endif //FUSION_PROFILER_H
