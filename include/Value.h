#pragma once
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <cassert>



class Value {
public:
  std::function<void(void)> backward = []() {};
  std::vector<std::shared_ptr<Value>> children;
  double data;
  double grad;
  std::string op;
  std::string label;
  explicit Value(double data = 0.0f, std::vector<std::shared_ptr<Value>> children = {},
                 std::string op = "", std::string label = "")
      : children(std::move(children)), data(data), grad(0.0), op(op),
        label(label) {}

  static std::shared_ptr<Value> make(double data = 0.0,
                                     std::vector<std::shared_ptr<Value>> children = {},
                                     std::string op = "", std::string label = "") {
    return std::make_shared<Value>(data, children, op, label);
  }
};

using Val = std::shared_ptr<Value>;

Val tanh(Val a);
Val exp(Val a);
void TopoSort(Val a, std::vector<Val> &sorted, std::unordered_map<Val, bool> &visited) ;
void backPropogate(Val self);