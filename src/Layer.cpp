#include "Layer.h"

ILayer::ILayer(size_t num_weights, size_t num_neurons){
    neurons = std::vector<Neuron>(num_neurons, nullptr);
    for(size_t i = 0; i < num_neurons; i++){
        neurons[i] = std::make_shared<INeuron>(num_weights);
    }
}

std::vector<Val> ILayer::operator()(std::vector<Val> x){
    std::vector<Val> out = std::vector<Val>(neurons.size());
    for(size_t i = 0; i< neurons.size(); i++){
        out[i] = (*neurons[i])(x);
    }
    return out;
}
