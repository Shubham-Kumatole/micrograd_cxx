#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
using namespace std;

class Value {
public:
  function<void(void)> backward = []() {};
  vector<shared_ptr<Value>> children;
  double data;
  double grad;
  const string op;
  string label;
  explicit Value(double data = 0.0f, vector<shared_ptr<Value>> children = {},
                 string op = "", string label = "")
      : children(std::move(children)), data(data), grad(0.0), op(op),
        label(label) {}

  static std::shared_ptr<Value> make(double data = 0.0,
                                     vector<shared_ptr<Value>> children = {},
                                     string op = "", string label = "") {
    return std::make_shared<Value>(data, children, op, label);
  }
};

using Val = std::shared_ptr<Value>;

Val operator+(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data + b->data, vector<Val>{a, b}, "+", "");
  out->backward = [a, b, out]() {
    a->grad += out->grad;
    b->grad += out->grad;
  };
  return out;
}

Val operator-(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data - b->data, vector<Val>{a, b}, "-", "");
  out->backward = [a, b, out]() {
    a->grad += out->grad;
    b->grad += -1.0 * out->grad;
  };
  return out;
}

Val operator-(Val a, double b) {
  auto b_prime = make_shared<Value>(b);
  return a - b_prime;
}

Val operator-(double b, Val a) {
  auto b_prime = make_shared<Value>(b);
  return b_prime - a;
}

Val operator+(Val a, double b) {
  auto b_prime = make_shared<Value>(b);
  return a + b_prime;
}

Val operator+(double b, Val a) {
  auto b_prime = make_shared<Value>(b);
  return b_prime + a;
}

Val operator*(Val a, Val b) {
  auto out =
      std::make_shared<Value>(a->data * b->data, vector<Val>{a, b}, "*", "");
  out->backward = [a, b, out]() {
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
  };
  return out;
}

Val operator^(Val a, double n) {
  auto out = std::make_shared<Value>((double)pow(a->data, n), vector<Val>{a},
                                     "**", "");
  out->backward = [a, n, out]() {
    a->grad += n * pow(a->data, n - 1) * out->grad;
  };
  return out;
}

Val operator^(Val a, int n) {
  auto out = std::make_shared<Value>((double)pow(a->data, n), vector<Val>{a},
                                     "**", "");
  out->backward = [a, n, out]() {
    a->grad += n * pow(a->data, n - 1) * out->grad;
  };
  return out;
}

Val operator/(Val a, Val b) { return a * (b ^ -1); }

Val operator+(int a, Val b) {
  auto other = make_shared<Value>(a);
  return other + b;
}

Val operator*(double a, Val b) {
  auto other = make_shared<Value>(a);
  return other * b;
}

Val operator*(int a, Val b) {
  auto other = make_shared<Value>(a);
  return other * b;
}

Val exp(Val a) {
  double d = exp(a->data);
  auto out = make_shared<Value>(d, vector<Val>{a}, "e(x)", "");
  out->backward = [a, d, out]() { a->grad += d * out->grad; };
  return out;
}

Val tanh(Val a) {
  double t = tanh(a->data);
  Val out = make_shared<Value>(t, vector<Val>{a}, "tanh(x)", "");
  out->backward = [a, t, out]() { a->grad += (1.0 - pow(t, 2)) * out->grad; };
  return out;
}

static void TopoSort(Val a, vector<Val> &sorted,
                     unordered_map<Val, bool> &visited) {
  if (visited.find(a) == visited.end()) {
    visited[a] = true;
    for (auto& child : a->children) {
      TopoSort(child, sorted, visited);
    }
    sorted.push_back(a);
  }
}

void backPropogate(Val self) {
  vector<Val> sorted;
  unordered_map<Val, bool> visited;
  TopoSort(self, sorted, visited);
  self->grad = 1.0;
  for (int i = sorted.size() - 1; i >= 0; i--) {
    Val& cur = sorted[i];
    cur->backward();
    /*printf("Value(Label : %s, data : %lf, grad : %lf, op : %s)\n",
           cur->label.c_str(), cur->data, cur->grad, cur->op.c_str());*/
  }
}

class Module{
  public:
    virtual vector<Val> parameters();
    void zeroGrad(){
      vector<Val> parameters = this->parameters();
      for(auto& x: parameters){
        x->grad = 0.0;
      }
    }
};