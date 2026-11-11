"""Small helpers around Hugging Face ESM-2 (same alphabet for every ESM-2 size)."""
import numpy as np
import torch

ESM_TOKS = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
            "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
            "<null_1>", "<mask>"]
TOK = {t: i for i, t in enumerate(ESM_TOKS)}
CLS, PAD, EOS, MASK = TOK["<cls>"], TOK["<pad>"], TOK["<eos>"], TOK["<mask>"]

MODEL_NAMES = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}


def encode(seq):
    """[cls] + residues + [eos]; residue number p sits at index p (because seq starts at residue 1)."""
    return [CLS] + [TOK[a] for a in seq] + [EOS]


def variant_ids(wt_seq, pos, mut):
    ids = encode(wt_seq)
    ids[pos] = TOK[mut]
    return ids


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_config():
    from transformers import EsmConfig
    return EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=1026, pad_token_id=PAD,
                     mask_token_id=MASK, position_embedding_type="rotary")


def load_model(name, masked_lm=False):
    """name: 8M/35M/150M/650M/3B, a full HF id, or 'tiny-random' (offline test model)."""
    from transformers import EsmModel, EsmForMaskedLM
    cls = EsmForMaskedLM if masked_lm else EsmModel
    if name == "tiny-random":
        torch.manual_seed(0)
        cfg = _tiny_config()
        return EsmForMaskedLM(cfg) if masked_lm else EsmModel(cfg, add_pooling_layer=False)
    hf = MODEL_NAMES.get(name, name)
    kw = {} if masked_lm else {"add_pooling_layer": False}
    model = cls.from_pretrained(hf, **kw)
    # sanity check: our hard-coded alphabet must match the tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf)
        assert tok.convert_tokens_to_ids("A") == TOK["A"] and tok.mask_token_id == MASK
    except Exception as e:  # pragma: no cover
        print("WARNING: tokenizer check failed:", e)
    return model


def batches(n, bs):
    for i in range(0, n, bs):
        yield np.arange(i, min(i + bs, n))
