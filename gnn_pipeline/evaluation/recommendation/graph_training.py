"""Validation-only pairwise training for temporal graph recommenders."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch

from gnn_pipeline.artifacts import sha256_file, write_json_atomic

from .baselines import TfidfRecommender
from .embeddings import EmbeddingTable
from .fusion import ValidationFusionEvaluator, deterministic_top_indices, fuse_scores
from .graph_models import TemporalGraphRanker
from .interfaces import RankedItem, Recommender
from .sampling import PairwiseSamples
from .temporal_graph import TemporalGraphSnapshot


TEMPORAL_MODEL_CHECKPOINT_VERSION = 2


@dataclass
class GraphTrainingResult:
    model: TemporalGraphRanker
    best_epoch: int
    best_validation: Dict[str, float]
    history: pd.DataFrame
    config: Dict

    def save(self, path: str, *, snapshot_ids: Mapping[int, str], sample_id: str) -> Dict:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": TEMPORAL_MODEL_CHECKPOINT_VERSION,
            "model_config": self.model.get_config(),
            "state_dict": self.model.state_dict(),
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "training_config": self.config,
            "snapshot_ids": {str(key): value for key, value in snapshot_ids.items()},
            "sample_id": sample_id,
        }
        torch.save(payload, output)
        manifest = {
            "checkpoint_version": TEMPORAL_MODEL_CHECKPOINT_VERSION,
            "checkpoint": output.name,
            "checkpoint_sha256": sha256_file(str(output)),
            "model_config": self.model.get_config(),
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "training_config": self.config,
            "snapshot_ids": payload["snapshot_ids"],
            "sample_id": sample_id,
        }
        write_json_atomic(str(output.with_suffix(".json")), manifest)
        return manifest


class TemporalGraphTrainer:
    def __init__(
        self,
        *,
        benchmark,
        embedding_table: EmbeddingTable,
        snapshots: Mapping[int, TemporalGraphSnapshot],
        pairwise_samples: PairwiseSamples,
        validation_snapshot: TemporalGraphSnapshot,
        fusion_evaluator: ValidationFusionEvaluator | None = None,
        device: str = "cpu",
    ):
        self.benchmark = benchmark
        self.embedding_table = embedding_table
        self.snapshots = dict(snapshots)
        self.samples = pairwise_samples
        self.validation_snapshot = validation_snapshot
        self.fusion_evaluator = fusion_evaluator or ValidationFusionEvaluator(benchmark)
        if self.fusion_evaluator.candidate_ids != validation_snapshot.paper_ids:
            raise ValueError("Fusion evaluator candidates do not match validation snapshot")
        self.device = torch.device(device)

    def _validation_metrics(
        self, model: TemporalGraphRanker, fusion_weights
    ) -> Dict[str, float]:
        model.eval()
        data = self.validation_snapshot.data.to(self.device)
        query_ids = self.fusion_evaluator.queries["query_id"].astype(str).tolist()
        query_embeddings = torch.from_numpy(
            self.embedding_table.lookup(query_ids)
        ).float().to(self.device)
        with torch.inference_mode():
            candidate_embeddings = model.encode_candidates(data)
            scores = model.full_scores(query_embeddings, candidate_embeddings)
        return self.fusion_evaluator.evaluate(scores, fusion_weights)

    def train(
        self,
        model: TemporalGraphRanker,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
        fusion_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> GraphTrainingResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model.to(self.device)
        # Initialize lazy heterogeneous projections before optimizer creation.
        first_snapshot = self.snapshots[sorted(self.snapshots)[0]]
        with torch.no_grad():
            model.encode_candidates(first_snapshot.data.to(self.device))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        grouped = {
            int(cutoff): group.reset_index(drop=True)
            for cutoff, group in self.samples.triples.groupby("snapshot_cutoff")
        }
        model.eval()
        best_validation = self._validation_metrics(model, fusion_weights)
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_fusion_weight = float(best_validation["fusion_weight"])
        stale_epochs = 0
        history = [{"epoch": 0, "train_loss": None, **best_validation}]

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            losses = []
            for cutoff_year, triples in sorted(grouped.items()):
                if cutoff_year not in self.snapshots:
                    raise ValueError(f"Missing training snapshot for cutoff {cutoff_year}")
                snapshot = self.snapshots[cutoff_year]
                data = snapshot.data.to(self.device)
                paper_to_idx = snapshot.paper_to_idx
                query_embeddings = torch.from_numpy(
                    self.embedding_table.lookup(triples["query_id"].astype(str).tolist())
                ).float().to(self.device)
                positive_idx = torch.tensor(
                    [paper_to_idx[str(value)] for value in triples["positive_id"]],
                    dtype=torch.long,
                    device=self.device,
                )
                negative_idx = torch.tensor(
                    [paper_to_idx[str(value)] for value in triples["negative_id"]],
                    dtype=torch.long,
                    device=self.device,
                )
                candidate_embeddings = model.encode_candidates(data)
                positive_scores = model.pair_scores(
                    query_embeddings, candidate_embeddings[positive_idx]
                )
                negative_scores = model.pair_scores(
                    query_embeddings, candidate_embeddings[negative_idx]
                )
                losses.append(
                    -torch.nn.functional.logsigmoid(
                        positive_scores - negative_scores
                    ).mean()
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            validation = self._validation_metrics(model, fusion_weights)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    **validation,
                }
            )
            score = (validation["ndcg@10"], validation["recall@20"])
            best_score = (
                best_validation["ndcg@10"],
                best_validation["recall@20"],
            )
            if score > best_score:
                best_validation = validation
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_fusion_weight = float(validation["fusion_weight"])
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("Training produced no validation checkpoint")
        model.load_state_dict(best_state)
        model.fusion_weight = best_fusion_weight
        model.eval()
        config = {
            **model.get_config(),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "seed": int(seed),
            "embedding_cache_key": self.embedding_table.manifest["cache_key"],
            "sample_id": self.samples.manifest["sample_id"],
            "fusion_weights": [float(value) for value in fusion_weights],
            "selected_fusion_weight": best_fusion_weight,
        }
        return GraphTrainingResult(
            model=model,
            best_epoch=best_epoch,
            best_validation=best_validation,
            history=pd.DataFrame(history),
            config=config,
        )


def load_temporal_graph_ranker(
    checkpoint_path: str,
    *,
    snapshot: TemporalGraphSnapshot,
    device: str = "cpu",
) -> tuple[TemporalGraphRanker, Dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("checkpoint_version") not in (1, TEMPORAL_MODEL_CHECKPOINT_VERSION):
        raise ValueError("Unsupported temporal graph model checkpoint")
    config = payload["model_config"]
    model = TemporalGraphRanker(
        architecture=config["architecture"],
        metadata=snapshot.data.metadata(),
        text_dim=config["text_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        hgt_heads=config["hgt_heads"],
        model_version=config.get("model_version", 1),
        fusion_weight=config.get(
            "fusion_weight",
            payload.get("training_config", {}).get("selected_fusion_weight", 1.0),
        ),
    ).to(device)
    with torch.no_grad():
        model.encode_candidates(snapshot.data.to(device))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


class FrozenGraphRecommender(Recommender):
    """Adapter for validation or one-time final full-corpus ranking."""

    def __init__(
        self,
        *,
        model: TemporalGraphRanker,
        snapshot: TemporalGraphSnapshot,
        embedding_table: EmbeddingTable,
        name: str,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.snapshot = snapshot
        self.embedding_table = embedding_table
        self.name = name
        self.device = torch.device(device)
        self._candidate_embeddings = None
        self.tfidf = TfidfRecommender()

    def get_config(self):
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "model_config": self.model.get_config(),
            "snapshot_id": self.snapshot.snapshot_id,
            "embedding_cache_key": self.embedding_table.manifest["cache_key"],
        }

    def _encode(self):
        if self._candidate_embeddings is None:
            with torch.inference_mode():
                self._candidate_embeddings = self.model.encode_candidates(
                    self.snapshot.data.to(self.device)
                )
        return self._candidate_embeddings

    def prepare(self, query, candidates, visible_citation_edges=None):
        self.tfidf.prepare(query, candidates, visible_citation_edges)
        if candidates["paper_id"].astype(str).tolist() != self.snapshot.paper_ids:
            raise ValueError("Preparation candidates do not match frozen graph snapshot")
        self._encode()

    def rank(
        self,
        query,
        candidates,
        top_k,
        seed,
        visible_citation_edges=None,
    ):
        candidate_ids = candidates["paper_id"].astype(str).tolist()
        if candidate_ids != self.snapshot.paper_ids:
            raise ValueError("Evaluation candidates do not match frozen graph snapshot")
        query_embedding = torch.from_numpy(
            self.embedding_table.lookup([str(query["query_id"])])
        ).float().to(self.device)
        with torch.inference_mode():
            neural_scores = self.model.full_scores(
                query_embedding, self._encode()
            )[0].cpu().numpy()
        lexical_scores = self.tfidf.score_candidates(query, candidates)
        scores = fuse_scores(
            neural_scores, lexical_scores, self.model.fusion_weight
        )
        top = deterministic_top_indices(scores, candidate_ids, top_k)[0]
        return [
            RankedItem(candidate_ids[index], float(scores[index])) for index in top
        ]
