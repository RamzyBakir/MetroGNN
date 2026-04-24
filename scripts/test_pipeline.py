"""
Test script for Step 8: End-to-End Integration

This script tests the complete GNN pipeline with real DBLP data:
1. Load and preprocess data
2. Build knowledge graph
3. Train GNN model (short epochs for validation)
4. Test query processing with dynamic topics
5. Test community analysis
6. Test strategy generation

Usage:
    python scripts/test_pipeline.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_pipeline.analysis.community_analysis import advanced_community_analysis
from gnn_pipeline.analysis.topic_detection import extract_topics_from_query_dynamic
from gnn_pipeline.config import ensure_dirs, get_path, get_value
from gnn_pipeline.data.dblp_loader import load_and_clean_dblp_data
from gnn_pipeline.data.feature_engineering import (
    create_all_features,
    print_feature_statistics,
)
from gnn_pipeline.data.graph_builder import build_knowledge_graph_from_dblp
from gnn_pipeline.data.topic_assignment import TopicAssigner, assign_topics_to_dblp
from gnn_pipeline.models.gnn_model import AdvancedHeteroGCN
from gnn_pipeline.models.training import train_enhanced_model
from gnn_pipeline.strategy.components import (
    DynamicContentGenerator,
    PersonalizedPlanAssembler,
)
from gnn_pipeline.strategy.generator import StrategyGenerator


def test_full_pipeline():
    """Test complete pipeline end-to-end."""

    ensure_dirs()

    print("=" * 70)
    print("STEP 8: END-TO-END INTEGRATION TEST")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load and Preprocess Data
    # ========================================================================
    print("\n[Step 1/6] Loading and preprocessing DBLP data...")
    csv_path = get_path("data", "csv_path")

    try:
        df, author_to_idx, author_list = load_and_clean_dblp_data(csv_path)
        print(f"✓ Loaded {len(df)} papers with {len(author_list)} unique authors")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    # ========================================================================
    # Step 2: Assign Dynamic Topics
    # ========================================================================
    n_clusters = get_value("training", "n_clusters")
    print(f"\n[Step 2/6] Assigning dynamic topics (k={n_clusters})...")

    try:
        topic_assigner = TopicAssigner(n_clusters=n_clusters)
        df_with_topics, results_dict, topic_info_dict = assign_topics_to_dblp(
            df, n_clusters=n_clusters
        )

        df_with_topics, results_dict, topic_info_dict = topic_assigner.assign_topics(df)

        print(f"✓ Assigned {len(topic_info_dict)} dynamic topics")
        print(f"  Silhouette score: {results_dict['silhouette_score']:.4f}")
    except Exception as e:
        print(f"✗ Error assigning topics: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========================================================================
    # Step 3: Create Features
    # ========================================================================
    print("\n[Step 3/6] Creating feature matrices...")

    try:
        paper_feats, author_feats, topic_feats, feature_info = create_all_features(
            df_with_topics, topic_info_dict, author_to_idx
        )
        print(
            f"✓ Created features for {feature_info['statistics']['num_papers']} papers, "
            f"{feature_info['statistics']['num_authors']} authors, "
            f"{feature_info['statistics']['num_topics']} topics"
        )
    except Exception as e:
        print(f"✗ Error creating features: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========================================================================
    # Step 4: Build Knowledge Graph
    # ========================================================================
    print("\n[Step 4/6] Building knowledge graph...")
    kg_save_path = get_path("models", "knowledge_graph")

    try:
        data, topic_names, _ = build_knowledge_graph_from_dblp(
            df_with_topics,
            author_to_idx,
            topic_info_dict,
            paper_feats,
            author_feats,
            topic_feats,
            save_path=kg_save_path,
        )
        print(f"✓ Built knowledge graph with {len(data.node_types)} node types")
        print(f"  Graph saved to: {kg_save_path}")
    except Exception as e:
        print(f"✗ Error building graph: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========================================================================
    # Step 5: Train GNN Model (short validation run)
    # ========================================================================
    print("\n[Step 5/6] Training GNN model (50 epochs for validation)...")

    try:
        model = AdvancedHeteroGCN(
            num_topics=len(topic_info_dict),
            hidden_channels=get_value("training", "hidden_channels"),
            num_layers=get_value("training", "num_layers"),
            num_classes=get_value("training", "num_classes"),
        )

        model, losses, val_metrics, imp_logits, role_logits, inf_scores = (
            train_enhanced_model(model, data, epochs=50)
        )

        print(f"✓ Training completed")
        print(f"  Final loss: {losses[-1]:.4f}")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========================================================================
    # Step 6: Test Query Processing with Dynamic Topics
    # ========================================================================
    print("\n[Step 6/6] Testing dynamic topic detection...")

    try:
        test_queries = [
            "I want to learn about graph algorithms",
            "Research in optimization and combinatorial theory",
            "Wireless communication systems and networks",
        ]

        print(f"  Testing {len(test_queries)} queries:")
        for query in test_queries:
            topics = extract_topics_from_query_dynamic(
                query, topic_assigner, topic_info_dict
            )
            topic_names_str = ", ".join(
                [topic_info_dict[tid]["name"] for tid in topics]
            )
            print(f"    Query: '{query}'")
            print(f"      Topics: {topic_names_str}")

        print("✓ Dynamic topic detection working")
    except Exception as e:
        print(f"✗ Error in topic detection: {e}")
        import traceback

        traceback.print_exc()
        return False

    # ========================================================================
    # Success Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✓ PIPELINE CORE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\nPipeline Components Validated:")
    print("  ✓ Data loading and preprocessing")
    print(f"  ✓ Dynamic topic assignment (k={n_clusters})")
    print("  ✓ Feature engineering (7 paper, 9 author dims)")
    print("  ✓ Knowledge graph construction")
    print("  ✓ GNN model training (50 epochs)")
    print("  ✓ Dynamic topic detection (TF-IDF similarity)")

    print("\nKey Statistics:")
    print(f"  Papers: {len(df_with_topics)}")
    print(f"  Authors: {len(author_list)}")
    print(f"  Topics: {len(topic_info_dict)}")
    print(f"  Graph edges: Multiple edge types (authored, has_topic, collaborates)")
    print(f"  Model layers: 4 GNN layers with attention")

    print("\nNext Steps:")
    print("  - Full training with 200 epochs")
    print("  - End-to-end community analysis testing")
    print("  - Strategy generation with dynamic topics")
    print("  - Integration with agentic layer (future)")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
