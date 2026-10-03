"""Landmark deep-learning papers the corpus must contain.

Why this exists alongside the cohort crawl: cohort sampling takes a few hundred
papers from each (category, year), but cs.LG alone publishes tens of thousands a
year. Sampling gives the statistical depth that field normalization needs - a
stable median to compare against - yet it catches any individual famous paper only
by luck. A deep-learning corpus missing ResNet and GPT-3 is not a credible one, and
an evaluation query asking for a paper the corpus doesn't hold is unanswerable by
construction: it would measure nothing but the gap in our own crawl.

So: cohort sampling for statistics, this list for landmark coverage.

These are stored as TITLES, not arXiv ids, deliberately. Ids written from memory are
easy to get subtly wrong and the error is invisible - a wrong-but-valid id silently
seeds the wrong paper. Titles get resolved through arXiv's exact-phrase title search
and the resolved title is then checked against what was asked for, so a bad match is
reported rather than quietly accepted.

This list doubles as the related-work spine for the project report.
"""

LANDMARK_TITLES: list[str] = [
    # --- Foundational architectures ---
    "Attention Is All You Need",
    "Deep Residual Learning for Image Recognition",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    "Generative Adversarial Networks",
    "Adam: A Method for Stochastic Optimization",
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
    "Layer Normalization",
    "Sequence to Sequence Learning with Neural Networks",
    "Neural Machine Translation by Jointly Learning to Align and Translate",
    "U-Net: Convolutional Networks for Biomedical Image Segmentation",
    # --- Language models ---
    "Language Models are Few-Shot Learners",
    "Training language models to follow instructions with human feedback",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "Scaling Laws for Neural Language Models",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    # --- Efficiency and adaptation ---
    "LoRA: Low-Rank Adaptation of Large Language Models",
    "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    "Distilling the Knowledge in a Neural Network",
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    # --- Retrieval and RAG (the project's own related work) ---
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Dense Passage Retrieval for Open-Domain Question Answering",
    "REALM: Retrieval-Augmented Language Model Pre-Training",
    "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "Corrective Retrieval Augmented Generation",
    "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
    "Precise Zero-Shot Dense Retrieval without Relevance Labels",
    "Passage Re-ranking with BERT",
    "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
    # --- Agents ---
    "ReAct: Synergizing Reasoning and Acting in Language Models",
    "Reflexion: Language Agents with Verbal Reinforcement Learning",
    "Toolformer: Language Models Can Teach Themselves to Use Tools",
    # --- Vision and multimodal ---
    "Learning Transferable Visual Models From Natural Language Supervision",
    "Denoising Diffusion Probabilistic Models",
    "High-Resolution Image Synthesis with Latent Diffusion Models",
    "Segment Anything",
    # --- Scholarly document representation (directly relevant to our ranker) ---
    "SPECTER: Document-level Representation Learning using Citation-informed Transformers",
    "Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings",
]
