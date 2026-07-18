"""Pairwise training for the no-graph dual-encoder architecture baseline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from gnn_pipeline.artifacts import sha256_file, write_json_atomic

from .baselines import TfidfRecommender
from .embeddings import EmbeddingTable
from .fusion import ValidationFusionEvaluator, deterministic_top_indices, fuse_scores
from .graph_models import TextDualEncoder
from .interfaces import RankedItem, Recommender
from .sampling import PairwiseSamples


@dataclass
class TextTrainingResult:
    model: TextDualEncoder
    best_epoch: int
    best_validation: Dict[str, float]
    config: Dict

    def save(self, path: str, *, embedding_cache_key: str, sample_id: str) -> Dict:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": self.model.model_version,
                "config": self.config,
                "state_dict": self.model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_validation": self.best_validation,
                "embedding_cache_key": embedding_cache_key,
                "sample_id": sample_id,
            },
            output,
        )
        manifest = {
            "checkpoint": output.name,
            "checkpoint_sha256": sha256_file(str(output)),
            "config": self.config,
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "embedding_cache_key": embedding_cache_key,
            "sample_id": sample_id,
        }
        write_json_atomic(str(output.with_suffix(".json")), manifest)
        return manifest


class TextDualEncoderTrainer:
    def __init__(
        self,
        *,
        benchmark,
        embedding_table: EmbeddingTable,
        samples: PairwiseSamples,
        fusion_evaluator: ValidationFusionEvaluator | None = None,
        device="cpu",
    ):
        self.benchmark = benchmark
        self.table = embedding_table
        self.samples = samples
        self.fusion_evaluator = fusion_evaluator or ValidationFusionEvaluator(benchmark)
        self.device = torch.device(device)

    def _validation(self, model: TextDualEncoder, fusion_weights):
        query_ids = self.fusion_evaluator.queries["query_id"].astype(str).tolist()
        candidate_ids = self.fusion_evaluator.candidate_ids
        with torch.inference_mode():
            candidate_embeddings = model.encode_candidates(
                torch.from_numpy(self.table.lookup(candidate_ids)).float().to(self.device)
            )
            query_embeddings = model.encode_queries(
                torch.from_numpy(self.table.lookup(query_ids)).float().to(self.device)
            )
            scores = query_embeddings @ candidate_embeddings.T
        return self.fusion_evaluator.evaluate(scores, fusion_weights)

    def train(
        self,
        *,
        hidden_dim=64,
        dropout=0.1,
        learning_rate=1e-3,
        max_epochs=100,
        patience=10,
        seed=42,
        model_version=2,
        fusion_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> TextTrainingResult:
        torch.manual_seed(seed)
        model = TextDualEncoder(
            self.table.dimension,
            hidden_dim,
            dropout,
            model_version=model_version,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        triples = self.samples.triples
        query = torch.from_numpy(self.table.lookup(triples["query_id"].astype(str))).float().to(self.device)
        positive = torch.from_numpy(self.table.lookup(triples["positive_id"].astype(str))).float().to(self.device)
        negative = torch.from_numpy(self.table.lookup(triples["negative_id"].astype(str))).float().to(self.device)

        model.eval()
        best = self._validation(model, fusion_weights)
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_fusion_weight = float(best["fusion_weight"])
        stale = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = -torch.nn.functional.logsigmoid(
                model.score(query, positive) - model.score(query, negative)
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            validation = self._validation(model, fusion_weights)
            if (validation["ndcg@10"], validation["recall@20"]) > (
                best["ndcg@10"], best["recall@20"]
            ):
                best = validation
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_fusion_weight = float(validation["fusion_weight"])
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        model.fusion_weight = best_fusion_weight
        model.eval()
        return TextTrainingResult(
            model=model,
            best_epoch=best_epoch,
            best_validation=best,
            config={
                "architecture": "text_dual_encoder",
                "model_version": int(model_version),
                "input_dim": self.table.dimension,
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "seed": int(seed),
                "fusion_weights": [float(value) for value in fusion_weights],
                "selected_fusion_weight": best_fusion_weight,
            },
        )


def load_text_dual_encoder(path: str, device="cpu") -> tuple[TextDualEncoder, Dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    model = TextDualEncoder(
        config["input_dim"],
        config["hidden_dim"],
        config["dropout"],
        model_version=config.get("model_version", 1),
        fusion_weight=config.get("selected_fusion_weight", 1.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


class FrozenTextDualEncoderRecommender(Recommender):
    def __init__(self, model, embedding_table, name="text_dual_encoder", device="cpu"):
        self.model = model.to(device).eval()
        self.table = embedding_table
        self.name = name
        self.device = torch.device(device)
        self._candidate_cache = {}
        self.tfidf = TfidfRecommender()

    def get_config(self):
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "embedding_cache_key": self.table.manifest["cache_key"],
        }

    def prepare(self, query, candidates, visible_citation_edges=None):
        year = int(query["query_year"])
        self.tfidf.prepare(query, candidates, visible_citation_edges)
        ids = candidates["paper_id"].astype(str).tolist()
        if year not in self._candidate_cache:
            with torch.inference_mode():
                embeddings = self.model.encode_candidates(
                    torch.from_numpy(self.table.lookup(ids)).float().to(self.device)
                )
            self._candidate_cache[year] = (ids, embeddings)
        elif self._candidate_cache[year][0] != ids:
            raise ValueError("Candidate corpus changed for cached dual-encoder index")

    def rank(self, query, candidates, top_k, seed, visible_citation_edges=None):
        year = int(query["query_year"])
        ids = candidates["paper_id"].astype(str).tolist()
        self.prepare(query, candidates, visible_citation_edges)
        candidate_embeddings = self._candidate_cache[year][1]
        with torch.inference_mode():
            query_embedding = self.model.encode_queries(
                torch.from_numpy(self.table.lookup([str(query["query_id"])]))
                .float()
                .to(self.device)
            )
            neural_scores = (query_embedding @ candidate_embeddings.T)[0].cpu().numpy()
        lexical_scores = self.tfidf.score_candidates(query, candidates)
        scores = fuse_scores(neural_scores, lexical_scores, self.model.fusion_weight)
        top = deterministic_top_indices(scores, ids, top_k)[0]
        return [RankedItem(ids[index], float(scores[index])) for index in top]
