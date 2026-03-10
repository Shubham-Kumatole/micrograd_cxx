#pragma once
#include "Module.h"
#include <random>
class INeuron : public Module {
    public:
    Val bias;
    std::vector<Val> weights = {};
    INeuron(size_t num_weights);
    Val operator()(std::vector<Val> x);
    static double randomDouble() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }
    const std::vector<Val>& parameters() override{
       std::vector<Val>* out = new std::vector<Val>(weights.size() + 1);
        (*out)[out->size()-1] = bias;
        for(size_t i = 0; i < weights.size(); i++){
            (*out)[i] = weights[i];
        }
        return *out;
    }
    static std::shared_ptr<INeuron> make(size_t num_weights = 0){
        return std::make_shared<INeuron>(num_weights);
    } 
};

using Neuron = std::shared_ptr<INeuron>;