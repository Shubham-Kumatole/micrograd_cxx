#pragma once
#include "Module.h"
#include "Neuron.h"

class ILayer: public Module {
public:
    std::vector<Neuron> neurons;
    ILayer(size_t num_weights, size_t num_neurons);
    std::vector<Val> parameters() override {
        std::vector<Val> out;
        for(auto& neuron : neurons){
            auto params = neuron->parameters();
            out.insert(out.end(), params.begin(), params.end());
        }
        return out;
    }   

    std::vector<Val> operator()(std::vector<Val> x);

    static std::shared_ptr<ILayer> make(size_t num_weights = 0, size_t num_neurons = 0){
        return std::make_shared<ILayer>(num_weights, num_neurons);
    } 
};


using Layer = std::shared_ptr<ILayer>;