#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class GeluBackward : public Node {
 public:
  GeluBackward(const Value& grad_output, const Value& input,
                     xla::int64_t approximate);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64_t approximate() const { return approximate_; }

 private:
  xla::int64_t approximate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
