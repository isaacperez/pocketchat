import torch

from pocketchat.model import Decomposer


def test_decomposer_output_shape_for_batch_embeddings() -> None:
    module = Decomposer(embedding_dim=8, num_components=4)
    x = torch.randn(3, 8)  # [B, D]

    out = module(x)

    assert out.shape == (3, 4, 8)


def test_decomposer_output_shape_for_hierarchical_embeddings() -> None:
    module = Decomposer(embedding_dim=6, num_components=3)
    x_hyp_components = torch.randn(2, 5, 6)  # [B, H, D]
    x_parts = torch.randn(2, 5, 7, 6)  # [B, H, P, D]

    out_components = module(x_hyp_components)
    out_parts = module(x_parts)

    assert out_components.shape == (2, 5, 3, 6)
    assert out_parts.shape == (2, 5, 7, 3, 6)


def test_decomposer_reduces_to_pure_residual_when_projection_is_zero() -> None:
    module = Decomposer(embedding_dim=5, num_components=2, bias=True)
    x = torch.randn(4, 5)

    with torch.no_grad():
        module.proj.weight.zero_()
        module.proj.bias.zero_()

    out = module(x)
    expected = x.unsqueeze(-2).expand(4, 2, 5)

    assert torch.allclose(out, expected, atol=1e-7)


def test_decomposer_validates_input_and_constructor_arguments() -> None:
    invalid_ctor_args = (
        {"embedding_dim": 0, "num_components": 2},
        {"embedding_dim": 8, "num_components": 0},
        {"embedding_dim": 8, "num_components": 2, "eps": 0.0},
    )
    for kwargs in invalid_ctor_args:
        try:
            Decomposer(**kwargs)
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")
        except ValueError:
            pass

    module = Decomposer(embedding_dim=4, num_components=2)
    bad_rank = torch.randn(4)  # [D]
    bad_last_dim = torch.randn(3, 5)
    bad_dtype = torch.randint(0, 10, (2, 4), dtype=torch.long)

    for bad_input, expected_exception in (
        (bad_rank, ValueError),
        (bad_last_dim, ValueError),
        (bad_dtype, TypeError),
    ):
        try:
            _ = module(bad_input)
            raise AssertionError("Expected input validation failure.")
        except expected_exception:
            pass


def test_decomposer_backprop_flows_to_input_and_parameters() -> None:
    module = Decomposer(embedding_dim=7, num_components=3)
    x = torch.randn(2, 4, 7, requires_grad=True)  # [B, H, D]

    out = module(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    assert module.proj.weight.grad is not None
    assert module.proj.weight.grad.abs().sum().item() > 0
    assert module.rms_norm.weight.grad is not None
    assert module.rms_norm.weight.grad.abs().sum().item() > 0


def test_decomposer_stack() -> None:
    batch_size = 2
    embedding_dim = 8
    num_components = 3
    num_parts = 4

    hypothesis_to_components = Decomposer(
        embedding_dim=embedding_dim,
        num_components=num_components,
    )
    components_to_parts = Decomposer(
        embedding_dim=embedding_dim,
        num_components=num_parts,
    )

    hypotheses = torch.randn(batch_size, embedding_dim, requires_grad=True)  # [B, D]
    components = hypothesis_to_components(hypotheses)  # [B, C, D]
    parts = components_to_parts(components)  # [B, C, P, D]

    assert components.shape == (batch_size, num_components, embedding_dim)
    assert parts.shape == (batch_size, num_components, num_parts, embedding_dim)

    loss = parts.pow(2).mean()
    loss.backward()

    assert hypotheses.grad is not None
    assert hypotheses.grad.abs().sum().item() > 0
    assert hypothesis_to_components.proj.weight.grad is not None
    assert hypothesis_to_components.proj.weight.grad.abs().sum().item() > 0
    assert components_to_parts.proj.weight.grad is not None
    assert components_to_parts.proj.weight.grad.abs().sum().item() > 0
