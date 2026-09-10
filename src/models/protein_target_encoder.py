"""Protein Target Sequence Encoder for AuditDDI (ESM-2 / Amino-Acid Language Modeling).

Encodes primary amino acid sequences of macromolecular protein targets (CYP enzymes,
receptors, kinases, transporters) into contextualized protein representations for
inductive Cold-Target DDI generalization.

Features:
1. ESM-2 Transformer integration (`facebook/esm2_t6_8M_UR50D`) when transformers is available.
2. Fast learned residue-embedding + 1D Conv fallback when transformers is absent.
3. Length-invariant mean pooling and projection to target hidden dimensions (default: 64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AMINO_ACID_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: i + 1 for i, aa in enumerate(AMINO_ACID_VOCAB)}  # 0 is padding


class ProteinTargetSequenceEncoder(nn.Module):
    """Encodes primary amino-acid sequences into biological target representations."""

    def __init__(
        self,
        output_dim: int = 64,
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        use_esm: bool = False,
        embedding_dim: int = 320,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.use_esm = use_esm
        self.esm_model = None
        self.esm_tokenizer = None
        # ESM-2 is frozen in this project.  Cache only its pooled backbone
        # output, not the trainable projection, so repeated drug sequences do
        # not trigger a transformer forward pass for every DDI pair batch.
        self._esm_backbone_cache: dict[str, torch.Tensor] = {}

        if use_esm:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
                self.esm_model = AutoModel.from_pretrained(esm_model_name)
                # Freeze ESM-2 backbone to prevent catastrophic forgetting
                for param in self.esm_model.parameters():
                    param.requires_grad = False
                self.esm_model.eval()
                embedding_dim = self.esm_model.config.hidden_size
            except Exception:
                self.esm_model = None
                self.esm_tokenizer = None
                self.use_esm = False

        if not self.use_esm:
            # Standalone amino-acid embedding + 1D convolution fallback
            self.aa_embedding = nn.Embedding(len(AMINO_ACID_VOCAB) + 2, 64, padding_idx=0)
            self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(in_channels=128, out_channels=embedding_dim, kernel_size=5, padding=2)
            self.norm = nn.BatchNorm1d(embedding_dim)

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _tokenize_sequences(self, sequences: list[str], max_len: int = 512, device: torch.device | None = None) -> torch.Tensor:
        """Tokenize amino acid strings into padded index tensor."""
        batch_tokens = []
        for seq in sequences:
            tokens = [AA_TO_IDX.get(c, 0) for c in seq.upper()[:max_len]]
            if not tokens:
                tokens = [0]
            batch_tokens.append(tokens)

        max_batch_len = max(len(t) for t in batch_tokens)
        padded = [t + [0] * (max_batch_len - len(t)) for t in batch_tokens]
        tensor = torch.tensor(padded, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def _esm_pool_sequences(
        self,
        sequences: list[str],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return frozen ESM backbone embeddings for a non-empty sequence batch."""
        if self.esm_model is None or self.esm_tokenizer is None:
            raise RuntimeError('ESM pooling requested before ESM-2 was loaded.')
        # Calling parent_model.train() recursively changes child modules to
        # train mode.  Frozen ESM must remain deterministic and dropout-free.
        self.esm_model.eval()
        inputs = self.esm_tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            return (
                (token_embeddings * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp(min=1e-9)
            )

    def precompute_esm_backbone_embeddings(
        self,
        sequences: list[str],
        *,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> int:
        """Cache unique frozen ESM embeddings and return newly cached count.

        The cache is intentionally not part of the checkpoint: it is a runtime
        acceleration for a fixed input catalogue, while the projection head
        remains trainable and is always applied at forward time.
        """
        if not self.use_esm or self.esm_model is None or self.esm_tokenizer is None:
            return 0
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if device is None:
            device = next(self.parameters()).device
        missing = list(dict.fromkeys(
            sequence.strip() for sequence in sequences
            if isinstance(sequence, str) and sequence.strip()
            and sequence.strip() not in self._esm_backbone_cache
        ))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            pooled = self._esm_pool_sequences(batch, device=device)
            for sequence, embedding in zip(batch, pooled):
                self._esm_backbone_cache[sequence] = embedding.detach().cpu()
        return len(missing)

    def forward(self, sequences: list[str], device: torch.device | None = None) -> torch.Tensor:
        """Encode a batch of amino acid sequences into output representations.

        Args:
            sequences: List of raw amino-acid strings (e.g., ['MALIPDL...', 'MGLEALV...']).
            device: Target torch device.

        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        if not sequences:
            return torch.zeros((0, self.output_dim), device=device)

        if device is None:
            device = next(self.parameters()).device

        if self.use_esm and self.esm_model is not None and self.esm_tokenizer is not None:
            normalized = [sequence.strip() if isinstance(sequence, str) else '' for sequence in sequences]
            uncached = list(dict.fromkeys(
                sequence for sequence in normalized
                if sequence and sequence not in self._esm_backbone_cache
            ))
            # A caller may use arbitrary inference sequences; populate missing
            # entries safely, while the benchmark pre-warms all known sequences.
            if uncached:
                self.precompute_esm_backbone_embeddings(uncached, device=device)
            hidden_size = int(self.esm_model.config.hidden_size)
            pooled = torch.stack([
                self._esm_backbone_cache[sequence].to(device)
                if sequence else torch.zeros(hidden_size, device=device)
                for sequence in normalized
            ])
        else:
            # Fallback 1D CNN over amino-acid embeddings
            tokens = self._tokenize_sequences(sequences, max_len=512, device=device)
            # tokens: (batch_size, seq_len)
            embedded = self.aa_embedding(tokens).transpose(1, 2)  # (batch_size, 64, seq_len)
            x = F.relu(self.conv1(embedded))
            x = F.relu(self.conv2(x))
            x = self.norm(x)
            # Global mean pool across sequence length
            mask = (tokens > 0).unsqueeze(1).float()  # (batch_size, 1, seq_len)
            pooled = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-9)

        return self.projection(pooled)
