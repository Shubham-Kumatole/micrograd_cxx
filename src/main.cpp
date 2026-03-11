
#include "MultiLayerPerceptron.h"
#include "Value.h"
#include <iostream>
int main(){
    MLP m = MLP(3, {4,4,1});
    const std::vector<Val>&  x = make_vector({2.0, 3.0, -1.0});
    auto res = m(x);
    for(auto &s:m.parameters()){
        std::cout << "Value(data : "<< s->data << ", grad : "<< s->grad << ")" << std::endl;
    }
    std::cout<< std::endl << "backpropogation\n" <<std::endl;
    backPropogate(res[0]);
    for(auto &s:m.parameters()){
        std::cout << "Value(data : "<< s->data << ", grad : "<< s->grad << ", op : " << s->op << ")" << std::endl;
    }
    return 0;
}