#include "Layer.cpp"

class MLP: public Module{
    public:
    vector<Layer> layers;
    MLP(size_t num_inputs, vector<size_t> layer_dimensions){
        layers = vector<Layer>(1 + layer_dimensions.size());
        if(layers.size() >= 2){
            layers[0] = make_shared<ILayer>(num_inputs, layer_dimensions[0]);
            size_t num_weights = layer_dimensions[0];
            for(size_t i = 1; i < layer_dimensions.size(); i++){
                layers[i] = make_shared<ILayer>(num_weights, layer_dimensions[i]);
                num_weights = layer_dimensions[i];
            }
        }
    }

    vector<Val> parameters() override{
        vector<Val> out = {};
        for(auto& layer: layers){
            vector<Val> params = layer->parameters();
            out.insert(end(out), begin(params), end(params));
        }
        return out;
    }

    vector<Val> operator()(vector<Val> x) {
        for(auto& layer : layers)
            x = (*layer)(std::move(x));
        return x;
    }
};