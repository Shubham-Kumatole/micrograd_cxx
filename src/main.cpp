
#include "MultiLayerPerceptron.h"
#include "Value.h"
#include <memory>

int main(){
    MLP m = MLP(3, {4,4,1});
    std::vector<std::vector<double>> xx = {
            {2.0, 3.0, -1.0},
            {3.0, -1.0, 0.5},
            {0.5, 1.0, 1.0},
            {1.0, 1.0, -1.0},
    };
    const std::vector<Val>& ys = make_vector({1.0, -1.0, -1.0, 1.0});
    std::vector<std::vector<Val>>  xs(xx.size());
    for(size_t i = 0; i < xx.size(); i++){
        xs[i] = make_vector(xx[i]);
    }
    for(int i = 0; i < 20; i++){
        Val loss = std::make_shared<Value>(0.0);
        for(size_t idx = 0; idx < xs.size(); idx++){
            auto ypred = m(xs[idx]);
            loss = loss + (ypred[0] - ys[idx])^2;
        }
        m.zeroGrad();
        backPropogate(loss);
        for(auto &p: m.parameters()){
            p->data += -0.01 * p->grad;
        }
        printf("loss = %lf\n", loss->data);
    }
    return 0;
}