
#include "MultiLayerPerceptron.h"
#include "Value.h"
#include "matplotlibcpp.h"
#include <algorithm>
#include <cmath>
#include <memory>

int main(){
    MLP m = MLP(3, {4,4,1});
    std::vector<std::vector<double>> xx = {
            {2.0, 3.0, -1.0},
            {3.0, -1.0, 0.5},
            {0.5, 1.0, 1.0},
            {1.0, 1.0, -1.0},
    };
    const std::vector<Val> ys = make_vector({1.0, -1.0, -1.0, 1.0});
    std::vector<std::vector<Val>>  xs(xx.size());
    for(size_t i = 0; i < xx.size(); i++){
        xs[i] = make_vector(xx[i]);
    }
    std::vector<double> losses(1000,0.0);
    double miny = HUGE_VALF, maxy = -1.0;
    for(int i = 0; i < 100; i++){
        double learning_rate = 0.01;
        Val loss = std::make_shared<Value>(0.0);
        std::vector<Val> ypred;
        for(size_t idx = 0; idx < xs.size(); idx++){
            ypred = m(xs[idx]);
            loss = loss + ((ypred[0] - ys[idx])^2);
            printf("\n Prediction : \n ypred = %lf, ys[idx] = %lf\n", ypred[0]->data, ys[idx]->data);
        }
        m.zeroGrad();
        backPropogate(loss);
        for(auto &p: m.parameters()){
            p->data -= learning_rate * p->grad;
        }
        losses[i] = loss->data; 
        miny = std::min(miny, loss->data);
        maxy = std::min(maxy, loss->data);
    }
    matplotlibcpp::backend("TkAgg");
    matplotlibcpp::xlim(0, 100);
    matplotlibcpp::ylim(0.0, 5.0 );
    matplotlibcpp::plot(losses);
    matplotlibcpp::show();
    return 0;
}