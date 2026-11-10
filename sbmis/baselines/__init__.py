"""Continuous-relaxation MIS baselines, reimplemented from the official code (docs/continuous_baselines.md).

    pcqo.pcqo   pCQO-MIS (Alkhouri et al., ICML 2025), from cpp_impl/pcqo_mis.cpp
    pqqa.pqqa   PQQA (Ichikawa & Arai, ICLR 2025), from the first release of src/qqa.py

Both run a batch of graphs at once, padded to a common size (common.pad_graphs).
"""
