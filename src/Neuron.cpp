#include "Neuron.h"

INeuron::INeuron(size_t num_weights){
    bias =std::make_shared<Value>(0.0);
    weights =std::vector<Val>(num_weights);
    for(auto w: weights){
        w =std::make_shared<Value>(randomDouble());
    }
}


Val INeuron::operator()(std::vector<Val> x){
    assert(x.size() == weights.size());
    double activation = bias->data; 
    for(size_t i = 0; i < x.size(); i++){
        activation += x[i]->data * weights[i]->data;
    }
    return tanh(std::make_shared<Value>(activation));
}
    



