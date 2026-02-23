import torch

from pocketchat.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from pocketchat.data import CharCorpus


def _write_text(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_from_file_builds_capped_frequency_vocab(tmp_path) -> None:
    text = "aaaaabbbbcccdde"
    path = _write_text(tmp_path, "corpus.txt", text)

    corpus = CharCorpus.from_file(path, vocab_size=7)  # 4 special + top 3 chars

    assert corpus.vocab_size == 7
    assert corpus.id_to_char[:4] == [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    assert corpus.id_to_char[4:] == ["a", "b", "c"]

    unk_id = corpus.unk_id
    assert corpus.data.dtype == torch.long
    assert corpus.data.numel() == len(text)
    assert corpus.data[-2:].tolist() == [unk_id, unk_id]


def test_vocab_size_is_bounded_by_unique_characters(tmp_path) -> None:
    path = _write_text(tmp_path, "small.txt", "abca")

    corpus = CharCorpus.from_file(path, vocab_size=100)

    assert corpus.vocab_size == 7  # 4 special + 3 unique chars
    assert set(corpus.id_to_char[4:]) == {"a", "b", "c"}


def test_encode_unknown_character_maps_to_unk(tmp_path) -> None:
    path = _write_text(tmp_path, "encode.txt", "aaaab")
    corpus = CharCorpus.from_file(path, vocab_size=5)  # only "a" retained
    unk_id = corpus.unk_id

    encoded = corpus.encode("abz")
    assert encoded.tolist() == [corpus.char_to_id["a"], unk_id, unk_id]


def test_decode_skips_special_tokens_and_marks_invalid_ids(tmp_path) -> None:
    path = _write_text(tmp_path, "decode.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)
    ids = torch.tensor(
        [
            corpus.bos_id,
            corpus.char_to_id["a"],
            corpus.eos_id,
            999,
        ],
        dtype=torch.long,
    )

    decoded = corpus.decode(ids)
    assert decoded == "a?"


def test_sample_batch_shape_dtype_and_device(tmp_path) -> None:
    path = _write_text(tmp_path, "batch.txt", "abcd" * 30)
    corpus = CharCorpus.from_file(path, vocab_size=10)

    batch = corpus.sample_batch(batch_size=8, seq_len=16, device=torch.device("cpu"))

    assert batch.shape == (8, 16)
    assert batch.dtype == torch.long
    assert batch.device.type == "cpu"


def test_sample_batch_raises_for_short_corpus(tmp_path) -> None:
    path = _write_text(tmp_path, "short.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)

    try:
        corpus.sample_batch(batch_size=2, seq_len=3, device=torch.device("cpu"))
        raise AssertionError("Expected ValueError for seq_len >= corpus length")
    except ValueError as exc:
        assert "must be greater than seq_len" in str(exc)


def test_from_file_validates_vocab_size_and_special_tokens(tmp_path) -> None:
    path = _write_text(tmp_path, "validations.txt", "hello")

    invalid_kwargs = (
        {"vocab_size": 0},
        {"vocab_size": 3},  # fewer than default special tokens
        {"vocab_size": 10, "special_tokens": (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)},
    )

    for kwargs in invalid_kwargs:
        try:
            CharCorpus.from_file(path, **kwargs)
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")
        except ValueError:
            pass


def test_special_token_id_properties(tmp_path) -> None:
    path = _write_text(tmp_path, "token_ids.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)

    assert corpus.pad_id == corpus.char_to_id[PAD_TOKEN]
    assert corpus.unk_id == corpus.char_to_id[UNK_TOKEN]
    assert corpus.bos_id == corpus.char_to_id[BOS_TOKEN]
    assert corpus.eos_id == corpus.char_to_id[EOS_TOKEN]
