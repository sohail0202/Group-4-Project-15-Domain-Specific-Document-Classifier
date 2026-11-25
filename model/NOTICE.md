# Ledger BERT checkpoint

This checkpoint modifies Google's `bert_uncased_L-2_H-128_A-2` pretrained weights.
Ledger training updates the last encoder output projection and LayerNorm and adds a classifier and a confidence temperature.
The bundled vocabulary comes from the same model revision.

The upstream checkpoint is available under Apache License 2.0.
See `APACHE-2.0.txt`.

- [Google BERT-Tiny model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)
- [Pinned original revision](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f)
- [Turc et al., 2019](https://arxiv.org/abs/1908.08962)
