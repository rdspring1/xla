#include "torch_xla/csrc/ops/gelu_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

namespace {

xla::XlaOp BuildTanhGeluGrad(xla::XlaOp grad_output, xla::XlaOp input) {
    const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    auto beta = createScalar(kBeta, shape, input);
    auto kappa = createScalar(0.044715, shape, input);
    auto one = createScalar(1, shape, input);
    auto two = createScalar(2, shape, input);
    auto three = createScalar(3, shape, input);
    auto half = createScalar(0.5, shape, input);

    auto inner = beta * (input + kappa * xla::Pow(input, three));
    auto tanh_inner = xla::Tanh(inner);

    auto left = half * input;
    auto right = one + tanh_inner;

    auto left_derivative = half * right;

    auto tanh_derivative = one - tanh_inner * tanh_inner;
    auto inner_derivative = beta * (one + three * kappa * xla::Pow(input, two));
    auto right_derivative = left * tanh_derivative * inner_derivative;

    return grad_output * (left_derivative + right_derivative);
}

xla::XlaOp BuildNoneGeluGrad(xla::XlaOp grad_output, xla::XlaOp input) {
    const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
    constexpr float kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    auto alpha = createScalar(kAlpha, shape, input);
    auto one = createScalar(1, shape, input);
    auto half = createScalar(0.5, shape, input);

    auto scratch = xla::Erf(input * createScalar(M_SQRT1_2, shape, input));
    auto dinput = xla::Exp(input * input * createScalar(-0.5, shape, input));
    return grad_output * (half * (one + scratch) + input * dinput * alpha);
}

xla::XlaOp BuildGeluGrad(xla::XlaOp grad_output, xla::XlaOp input, xla::int64_t approximate) {
  const int64_t kTanh = 1;
  if (approximate == kTanh) {
    return BuildTanhGeluGrad(grad_output, input);
  } else {
    return BuildNoneGeluGrad(grad_output, input);
  }
}

} // namespace

namespace ir {
namespace ops {

GeluBackward::GeluBackward(const Value& grad_output,
                                       const Value& input, xla::int64_t approximate)
    : Node(ir::OpKind(at::aten::gelu_backward),
           {grad_output, input}, grad_output.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(approximate)),
      approximate_(approximate) {}

NodePtr GeluBackward::Clone(OpList operands) const {
  return MakeNode<GeluBackward>(operands.at(0), operands.at(1), approximate_);
}

xla::XlaOp createScalar(float value, xla::Shape& shape, xla::XlaOp input) {
  return XlaHelpers::ScalarValue<float>(value, shape.element_type(),
                                                   input.builder());
}

XlaOpVector GeluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildGeluGrad(grad_output, input, approximate_);
  return ReturnOp(grad_input, loctx);
}

std::string GeluBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", approximate=" << approximate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
