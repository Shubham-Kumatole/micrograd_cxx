#include "Neuron.h"

INeuron::INeuron(size_t num_weights){
    bias =std::make_shared<Value>(0.0);
    weights =std::vector<Val>(num_weights);
    for(auto &w: weights){
        w =std::make_shared<Value>(randomDouble());
    }
}


Val INeuron::operator()(std::vector<Val> x){
    assert(x.size() == this->weights.size());
    Val activation = std::make_shared<Value>(0.0);
    for(size_t i = 0; i < x.size(); i++){
        activation =  activation + weights[i] * x[i];
    }
    return tanh(activation + bias);
}
    



