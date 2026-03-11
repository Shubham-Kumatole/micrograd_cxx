#pragma once
#include "Value.h"

class Module{
  public:
    virtual ~Module() = default;
    virtual std::vector<Val> parameters() = 0;
    void zeroGrad(){
        for(auto& x : parameters()){
            x->grad = 0.0;
        }
    }
};