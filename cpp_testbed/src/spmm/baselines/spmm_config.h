//
// Created by lwilkinson on 6/6/22.
//

#ifndef DNN_SPMM_BENCH_SPMM_CONFIG_H
#define DNN_SPMM_BENCH_SPMM_CONFIG_H

#include <map>
#include <iostream>
#include <sstream>

#define REGISTER_PARAMS_BEGIN(stuct_name) stuct_name()  { param_offsets = {
#define REGISTER_PARAM(str, param)                      {str, (int*)&this->param - (int*)this }
#define REGISTER_PARAMS_END()                           };}

struct ConfigBase {
protected:
    std::map<std::string, size_t> param_offsets;
public:
    void setVal(std::string key, int val) {
        if (param_offsets.count(key)) {
            *((int*)this + param_offsets[key]) = val;
        } else {
            std::cerr << "Setting " << key << " not supported on this config" << std::endl;
        }
    };

    int getVal(std::string key) {
        if (param_offsets.count(key)) {
            return *((int*)this + param_offsets[key]);
        } else {
            std::cerr << "Getting " << key << " not supported on this config" << std::endl;
            return 0;
        }
    };

    void print() {
        for (auto& p : param_offsets) {
            std::cout << p.first << ": " << *((int*)this + p.second) << std::endl;
        }
    }

    std::string rep() {
        std::stringstream ss;
        for (auto& p : param_offsets) {
            ss << p.first << ":" << *((int*)this + p.second) << "|";
        }
        std::string rep_ = ss.str();
        rep_.pop_back();

        return rep_;
    }
};

#endif //DNN_SPMM_BENCH_SPMM_CONFIG_H
