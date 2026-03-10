#include "MultiLayerPerceptron.h"


MLP::MLP(size_t num_inputs, std::vector<size_t> layer_dimensions){
    layers = std::vector<Layer>(1 + layer_dimensions.size());
    if(layers.size() >= 2){
        layers[0] = std::make_shared<ILayer>(num_inputs, layer_dimensions[0]);
        size_t num_weights = layer_dimensions[0];
        for(size_t i = 1; i < layer_dimensions.size(); i++){
            layers[i] = std::make_shared<ILayer>(num_weights, layer_dimensions[i]);
            num_weights = layer_dimensions[i];
        }
    }
}

std::vector<Val> MLP::operator()(std::vector<Val> x) {
    for(auto& layer : layers)
        x = (*layer)(std::move(x));
    return x;
}
