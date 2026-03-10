#pragma once
#include "Module.h"
#include "Neuron.h"

class ILayer: public Module {
public:
    std::vector<Neuron> neurons;
    ILayer(size_t num_weights, size_t num_neurons);
    const std::vector<Val>& parameters() override{
        std::vector<Val>* out = new std::vector<Val>();
        for(auto& neuron: neurons){
            const std::vector<Val>& params = neuron->parameters();
            (*out).insert(end(*out), begin(params), end(params));
        }
        return *out;
    }

    std::vector<Val> operator()(std::vector<Val> x);

    static std::shared_ptr<ILayer> make(size_t num_weights = 0, size_t num_neurons = 0){
        return std::make_shared<ILayer>(num_weights, num_neurons);
    } 
};


using Layer = std::shared_ptr<ILayer>;