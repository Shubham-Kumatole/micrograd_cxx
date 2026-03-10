#include "Neuron.cpp"

class ILayer: public Module{
public:
    vector<Neuron> neurons;
    ILayer(size_t num_weights, size_t num_neurons){
        neurons = vector<Neuron>(num_neurons, nullptr);
        for(size_t i = 0; i < num_neurons; i++){
            neurons[i] = make_shared<INeuron>(num_weights);
        }
    }
    vector<Val> parameters() override{
        vector<Val> out = {};
        for(auto& neuron: neurons){
            vector<Val> params = neuron->parameters();
            out.insert(end(out), begin(params), end(params));
        }
        return out;
    }

    vector<Val> operator()(vector<Val> x){
        vector<Val> out = vector<Val>(neurons.size());
        for(size_t i = 0; i< neurons.size(); i++){
            out[i] = (*neurons[i])(x);
        }
        return out;
    }

    static std::shared_ptr<ILayer> make(size_t num_weights = 0, size_t num_neurons = 0){
        return std::make_shared<ILayer>(num_weights, num_neurons);
    } 
};

using Layer = shared_ptr<ILayer>;