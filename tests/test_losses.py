import torch
import torch.nn.functional as F

from pocketchat.losses import next_char_cross_entropy_loss


def test_next_char_cross_entropy_loss_with_targets_per_position_uses_last_position_by_default() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.2, -1.0], [0.1, 1.2, 0.3]],
            [[-0.2, 2.0, 0.7], [1.5, 0.2, -0.4]],
        ],
        dtype=torch.float32,
    )  # [B=2, N=2, V=3]
    targets = torch.tensor(
        [
            [[0, 1], [2, 0], [1, 2]],
            [[2, 0], [1, 1], [0, 2]],
        ],
        dtype=torch.long,
    )  # [B=2, L=3, N=2]

    loss = next_char_cross_entropy_loss(logits, targets, ignore_index=-100)
    expected = F.cross_entropy(logits.reshape(-1, 3), targets[:, -1, :].reshape(-1), ignore_index=-100)

    assert torch.allclose(loss, expected, atol=1e-8)


def test_next_char_cross_entropy_loss_uses_lengths_to_select_last_valid_position() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.5, -0.5], [0.1, 1.8, 0.2]],
            [[0.3, 1.1, 0.4], [2.2, -0.2, 0.0]],
        ],
        dtype=torch.float32,
    )  # [B=2, N=2, V=3]
    targets = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 0], [1, 1]],
            [[2, 1], [0, 2], [1, 0], [2, 2]],
        ],
        dtype=torch.long,
    )  # [B=2, L=4, N=2]
    lengths = torch.tensor([2, 4], dtype=torch.long)

    loss = next_char_cross_entropy_loss(logits, targets, ignore_index=-100, lengths=lengths)
    expected_targets = torch.stack([targets[0, 1], targets[1, 3]], dim=0)  # lengths-1
    expected = F.cross_entropy(logits.reshape(-1, 3), expected_targets.reshape(-1), ignore_index=-100)

    assert torch.allclose(loss, expected, atol=1e-8)


def test_next_char_cross_entropy_loss_supports_direct_targets_shape() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor(
        [
            [1, 2, 4],
            [0, 3, 1],
        ],
        dtype=torch.long,
    )  # [B, N]

    loss = next_char_cross_entropy_loss(logits, targets, ignore_index=-100)
    expected = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), ignore_index=-100)

    assert torch.allclose(loss, expected, atol=1e-8)


def test_next_char_cross_entropy_loss_uses_ignore_index() -> None:
    pad_id = 0
    logits = torch.tensor(
        [
            [[2.0, -1.0, 0.3], [1.0, 0.2, -0.7]],
            [[-0.1, 1.8, 0.4], [0.6, -0.2, 1.5]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [2, pad_id],
            [1, 2],
        ],
        dtype=torch.long,
    )

    loss = next_char_cross_entropy_loss(logits, targets, ignore_index=pad_id)

    flat_logits = logits.reshape(-1, 3)
    flat_targets = targets.reshape(-1)
    mask = flat_targets != pad_id
    expected = F.cross_entropy(flat_logits[mask], flat_targets[mask])

    assert torch.allclose(loss, expected, atol=1e-8)


def test_next_char_cross_entropy_loss_validates_inputs() -> None:
    bad_logits = (
        torch.randn(2, 3),  # rank mismatch
        torch.ones(2, 3, 4, dtype=torch.long),  # non-float logits
    )
    good_targets = torch.randint(0, 4, (2, 3), dtype=torch.long)

    for logits in bad_logits:
        try:
            next_char_cross_entropy_loss(logits, good_targets, ignore_index=-100)
            raise AssertionError("Expected validation error for invalid logits.")
        except (ValueError, TypeError):
            pass

    logits = torch.randn(2, 3, 4)
    bad_targets = (
        torch.randint(0, 4, (2, 2), dtype=torch.long),  # shape mismatch [B, N]
        torch.randint(0, 4, (2, 2, 2, 2), dtype=torch.long),  # rank mismatch
        torch.randn(2, 3),  # non-integer targets
    )
    for targets in bad_targets:
        try:
            next_char_cross_entropy_loss(logits, targets, ignore_index=-100)
            raise AssertionError("Expected validation error for invalid targets.")
        except (ValueError, TypeError):
            pass


def test_next_char_cross_entropy_loss_validates_lengths_argument() -> None:
    logits = torch.randn(2, 3, 4)
    targets = torch.randint(0, 4, (2, 5, 3), dtype=torch.long)

    invalid_lengths = (
        torch.tensor([[2, 3]], dtype=torch.long),  # rank mismatch
        torch.tensor([2, 3, 4], dtype=torch.long),  # batch mismatch
        torch.tensor([0, 2], dtype=torch.long),  # invalid lower bound
        torch.tensor([2, 6], dtype=torch.long),  # invalid upper bound
        torch.tensor([2.0, 3.0], dtype=torch.float32),  # non-integer dtype
    )

    for lengths in invalid_lengths:
        try:
            next_char_cross_entropy_loss(
                logits,
                targets,
                ignore_index=-100,
                lengths=lengths,
            )
            raise AssertionError("Expected validation error for invalid lengths.")
        except (ValueError, TypeError):
            pass


def test_next_char_cross_entropy_loss_applies_horizon_decay_weights() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.2, 2.0], [1.5, 0.5]],
        ],
        dtype=torch.float32,
    )  # [B=1, N=3, V=2]
    targets = torch.tensor([[0, 1, 1]], dtype=torch.long)  # [B=1, N=3]

    decay = 0.5
    loss = next_char_cross_entropy_loss(
        logits,
        targets,
        ignore_index=-100,
        horizon_decay=decay,
    )

    token_losses = F.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
        reduction="none",
    ).reshape(1, 3)
    weights = torch.tensor([1.0, decay, decay**2], dtype=torch.float32).reshape(1, 3)
    expected = (token_losses * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected, atol=1e-8)


def test_next_char_cross_entropy_loss_validates_horizon_decay() -> None:
    logits = torch.randn(2, 3, 4)
    targets = torch.randint(0, 4, (2, 3), dtype=torch.long)

    for bad_decay in (0.0, -0.1):
        try:
            next_char_cross_entropy_loss(
                logits,
                targets,
                ignore_index=-100,
                horizon_decay=bad_decay,
            )
            raise AssertionError("Expected validation error for invalid horizon_decay.")
        except ValueError:
            pass
