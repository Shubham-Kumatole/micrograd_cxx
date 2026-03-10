#pragma once
#include "Value.h"

class Module{
  public:
    virtual ~Module() = default;
    virtual const std::vector<Val>& parameters() = 0;
    void zeroGrad(){
      std::vector<Val> parameters = this->parameters();
      for(auto& x: parameters){
        x->grad = 0.0;
      }
    }
};