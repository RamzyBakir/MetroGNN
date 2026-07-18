"""Pairwise linear and MLP rerankers over leakage-safe temporal features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, TemporalFeatureStore
from .interfaces import RankedItem, Recommender
from gnn_pipeline.artifacts import sha256_file, write_json_atomic

from .sampling import PairwiseSamples


class _FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class PairwiseFeatureRecommender(Recommender):
    """Train a linear or nonlinear scorer from positive/unjudged pairs."""

    def __init__(
        self,
        feature_store: TemporalFeatureStore,
        samples: PairwiseSamples,
        *,
        architecture: str = "linear",
        hidden_dim: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        linear_c: float = 1.0,
        epochs: int = 100,
        seed: int = 42,
        name: str | None = None,
    ):
        if architecture not in {"linear", "mlp"}:
            raise ValueError("Feature architecture must be 'linear' or 'mlp'")
        self.feature_store = feature_store
        self.samples = samples
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.linear_c = float(linear_c)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.name = name or f"feature_{architecture}"
        self.scaler = StandardScaler()
        self.model = None
        self._fitted = False
        self._frozen = False

    def get_config(self):
        return {
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__name__}",
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "linear_c": self.linear_c,
            "epochs": self.epochs,
            "seed": self.seed,
            "feature_names": list(FEATURE_NAMES),
            "sample_id": self.samples.manifest["sample_id"],
            "embedding_cache_key": self.feature_store.embedding_table.manifest["cache_key"]
            if self.feature_store.embedding_table is not None
            else None,
            "enabled_features": sorted(self.feature_store.enabled_features),
        }

    def _training_matrices(self, training_data) -> Tuple[np.ndarray, np.ndarray]:
        queries = {
            str(row.query_id): pd.Series(row._asdict())
            for row in training_data.train_queries.itertuples(index=False)
        }
        positive_rows = []
        negative_rows = []
        for query_id, triples in self.samples.triples.groupby("query_id"):
            query = queries[str(query_id)]
            candidates = self.feature_store.benchmark.candidates_for_query(query)
            visible = self.feature_store.benchmark.visible_citation_edges(query)
            matrix = self.feature_store.candidate_features(
                query, candidates, visible_citation_edges=visible
            )
            lookup = {
                paper_id: index
                for index, paper_id in enumerate(candidates["paper_id"].astype(str))
            }
            positive_rows.extend(matrix[lookup[str(value)]] for value in triples["positive_id"])
            negative_rows.extend(matrix[lookup[str(value)]] for value in triples["negative_id"])
        return np.asarray(positive_rows), np.asarray(negative_rows)

    def fit(self, training_data):
        if self._frozen:
            return self
        positive, negative = self._training_matrices(training_data)
        stacked = np.vstack([positive, negative])
        self.scaler.fit(stacked)
        positive = self.scaler.transform(positive).astype(np.float32)
        negative = self.scaler.transform(negative).astype(np.float32)

        if self.architecture == "linear":
            differences = positive - negative
            x = np.vstack([differences, -differences])
            y = np.concatenate([np.ones(len(differences)), np.zeros(len(differences))])
            self.model = LogisticRegression(
                C=self.linear_c,
                random_state=self.seed,
                max_iter=1000,
            ).fit(x, y)
        else:
            torch.manual_seed(self.seed)
            self.model = _FeatureMLP(len(FEATURE_NAMES), self.hidden_dim, self.dropout)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
            )
            positive_tensor = torch.from_numpy(positive)
            negative_tensor = torch.from_numpy(negative)
            self.model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                difference = self.model(positive_tensor) - self.model(negative_tensor)
                loss = -torch.nn.functional.logsigmoid(difference).mean()
                loss.backward()
                optimizer.step()
            self.model.eval()
        self._fitted = True
        return self

    def _score(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features).astype(np.float32)
        if self.architecture == "linear":
            return self.model.decision_function(transformed)
        with torch.inference_mode():
            return self.model(torch.from_numpy(transformed)).numpy()

    def prepare(self, query, candidates, visible_citation_edges=None):
        if visible_citation_edges is None:
            raise ValueError("Feature recommender requires cutoff-visible citation edges")
        self.feature_store.candidate_features(
            query, candidates, visible_citation_edges=visible_citation_edges
        )

    def rank(
        self,
        query,
        candidates,
        top_k,
        seed,
        visible_citation_edges=None,
    ):
        if not self._fitted:
            raise RuntimeError("Feature recommender has not been fitted")
        if visible_citation_edges is None:
            raise ValueError("Feature recommender requires cutoff-visible citation edges")
        features = self.feature_store.candidate_features(
            query, candidates, visible_citation_edges=visible_citation_edges
        )
        scores = self._score(features)
        frame = pd.DataFrame(
            {
                "paper_id": candidates["paper_id"].astype(str).tolist(),
                "score": scores,
            }
        ).sort_values(["score", "paper_id"], ascending=[False, True], kind="mergesort")
        return [
            RankedItem(str(row.paper_id), float(row.score))
            for row in frame.head(top_k).itertuples(index=False)
        ]

    def save(self, path: str) -> Dict:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted feature recommender")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": 1,
            "config": self.get_config(),
            "scaler_mean": torch.as_tensor(self.scaler.mean_),
            "scaler_scale": torch.as_tensor(self.scaler.scale_),
            "scaler_var": torch.as_tensor(self.scaler.var_),
            "scaler_n_samples_seen": int(self.scaler.n_samples_seen_),
        }
        if self.architecture == "linear":
            payload.update(
                {
                    "linear_coef": torch.as_tensor(self.model.coef_),
                    "linear_intercept": torch.as_tensor(self.model.intercept_),
                    "linear_classes": torch.as_tensor(self.model.classes_),
                }
            )
        else:
            payload["state_dict"] = self.model.state_dict()
        torch.save(payload, output)
        manifest = {
            "checkpoint": output.name,
            "checkpoint_sha256": sha256_file(str(output)),
            "config": self.get_config(),
        }
        write_json_atomic(str(output.with_suffix(".json")), manifest)
        return manifest

    @classmethod
    def load(
        cls,
        path: str,
        *,
        feature_store: TemporalFeatureStore,
        samples: PairwiseSamples,
    ) -> "PairwiseFeatureRecommender":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload["config"]
        current_cache_key = (
            feature_store.embedding_table.manifest["cache_key"]
            if feature_store.embedding_table is not None
            else None
        )
        if config.get("embedding_cache_key") != current_cache_key:
            raise ValueError("Feature checkpoint embedding cache mismatch")
        if config.get("sample_id") != samples.manifest.get("sample_id"):
            raise ValueError("Feature checkpoint pairwise sample mismatch")
        if set(config.get("enabled_features", FEATURE_NAMES)) != feature_store.enabled_features:
            raise ValueError("Feature checkpoint enabled-feature set mismatch")
        model = cls(
            feature_store,
            samples,
            architecture=config["architecture"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            linear_c=config.get("linear_c", 1.0),
            epochs=config["epochs"],
            seed=config["seed"],
            name=config["name"],
        )
        model.scaler.mean_ = payload["scaler_mean"].numpy()
        model.scaler.scale_ = payload["scaler_scale"].numpy()
        model.scaler.var_ = payload["scaler_var"].numpy()
        model.scaler.n_samples_seen_ = payload["scaler_n_samples_seen"]
        model.scaler.n_features_in_ = len(FEATURE_NAMES)
        if model.architecture == "linear":
            classifier = LogisticRegression()
            classifier.coef_ = payload["linear_coef"].numpy()
            classifier.intercept_ = payload["linear_intercept"].numpy()
            classifier.classes_ = payload["linear_classes"].numpy()
            classifier.n_features_in_ = len(FEATURE_NAMES)
            model.model = classifier
        else:
            network = _FeatureMLP(len(FEATURE_NAMES), model.hidden_dim, model.dropout)
            network.load_state_dict(payload["state_dict"])
            network.eval()
            model.model = network
        model._fitted = True
        model._frozen = True
        return model
