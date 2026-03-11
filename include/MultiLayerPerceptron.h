#pragma once
#include "Module.h"
#include "Layer.h"

class MLP: public Module {
public:
    std::vector<Layer> layers;
    MLP(size_t num_inputs, std::vector<size_t> layer_dimensions);
    std::vector<Val> parameters() override{
        std::vector<Val> out;
        for(auto& layer: layers){
            std::vector<Val> params = layer->parameters();
            out.insert(end(out), begin(params), end(params));
        }
        return out;
    }
    std::vector<Val> operator()(std::vector<Val> x);
};