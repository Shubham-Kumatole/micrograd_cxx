#include "value.cpp"

class INeuron : public Module{
    public:
    Val bias;
    vector<Val> weights = {};
    INeuron(size_t num_weights){
        bias = make_shared<Value>(0.0);
        weights = vector<Val>(num_weights);
        for(auto w: weights){
            w = make_shared<Value>(randomDouble());
        }
    }


    Val operator()(vector<Val> x){
        assert(x.size() == weights.size());
        double activation = bias->data; 
        for(size_t i = 0; i < x.size(); i++){
            activation += x[i]->data * weights[i]->data;
        }
        return tanh(make_shared<Value>(activation));
    }
    
    vector<Val> parameters() override{
        vector<Val> out = vector<Val>(weights.size() + 1);
        out[out.size()-1] = bias;
        for(size_t i = 0; i < weights.size(); i++){
            out[i] = weights[i];
        }
        return out;
    }

    static std::shared_ptr<INeuron> make(size_t num_weights = 0){
        return std::make_shared<INeuron>(num_weights);
    } 

    double randomDouble() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }
};

using Neuron = shared_ptr<INeuron>;

