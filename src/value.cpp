#include "Value.h"
#include <cmath>
#include <memory>
#include <vector>

Val operator+(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data + b->data, std::vector<Val>{a, b}, "+", "");
  out->backward = [a, b, out]() {
    a->grad += out->grad;
    b->grad += out->grad;
  };
  return out;
}

Val operator-(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data - b->data, std::vector<Val>{a, b}, "-", "");
  out->backward = [a, b, out]() {
    a->grad += out->grad;
    b->grad += -1.0 * out->grad;
  };
  return out;
}

Val operator-(Val a, double b) {
  auto b_prime = std::make_shared<Value>(b);
  return a - b_prime;
}

Val operator-(double b, Val a) {
  auto b_prime = std::make_shared<Value>(b);
  return b_prime - a;
}

Val operator+(Val a, double b) {
  auto b_prime = std::make_shared<Value>(b);
  return a + b_prime;
}

Val operator+(double b, Val a) {
  auto b_prime = std::make_shared<Value>(b);
  return b_prime + a;
}

Val operator*(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data * b->data, std::vector<Val>{a, b}, "*", "");
  out->backward = [a, b, out]() {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
  };
  return out;
}

Val operator^(Val a, double n) {
  auto out = std::make_shared<Value>((double)pow(a->data, n), std::vector<Val>{a},
                                     "**", "");
  out->backward = [a, n, out]() {
    a->grad += n * pow(a->data, n - 1) * out->grad;
  };
  return out;
}

Val operator^(Val a, int n) {
  auto out = std::make_shared<Value>((double)pow(a->data, n), std::vector<Val>{a},
                                     "**", "");
  out->backward = [a, n, out]() {
    a->grad += n * pow(a->data, n - 1) * out->grad;
  };
  return out;
}

Val operator/(Val a, Val b) { return a * (b ^ -1); }

Val operator+(int a, Val b) {
  auto other = std::make_shared<Value>(a);
  return other + b;
}

Val operator*(double a, Val b) {
  auto other = std::make_shared<Value>(a);
  return other * b;
}

Val operator*(int a, Val b) {
  auto other = std::make_shared<Value>(a);
  return other * b;
}

Val exp(Val a) {
  double d = std::exp(a->data);
  auto out = std::make_shared<Value>(d, std::vector<Val>{a}, "e(x)", "");
  out->backward = [a, d, out]() { a->grad += d * out->grad; };
  return out;
}

Val tanh(Val a) {
  double t = std::tanh(a->data);
  Val out = std::make_shared<Value>(t, std::vector<Val>{a}, "tanh(x)", "");
  out->backward = [a, t, out]() { a->grad += (1.0 - pow(t, 2)) * out->grad; };
  return out;
}

std::vector<Val> make_vector(std::vector<double> x){
  std::vector<Val> out;
  for(auto &a: x){
    out.push_back(std::make_shared<Value>(a)); 
  }
  return out;
}

void TopoSort(Val a, std::vector<Val> &sorted,
                     std::unordered_map<Val, bool> &visited) {
  if (visited.find(a) == visited.end()) {
    visited[a] = true;
    for (auto& child : a->children) {
      TopoSort(child, sorted, visited);
    }
    sorted.push_back(a);
  }
}

void backPropogate(Val self) {
  std::vector<Val> sorted;
  std::unordered_map<Val, bool> visited;
  TopoSort(self, sorted, visited);
  self->grad = 1.0;
  for (int i = sorted.size() - 1; i >= 0; i--) {
    Val& cur = sorted[i];
    cur->backward();
    /*printf("Value(Label : %s, data : %lf, grad : %lf, op : %s)\n",
           cur->label.c_str(), cur->data, cur->grad, cur->op.c_str());*/
  }
}
