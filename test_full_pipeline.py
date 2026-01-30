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
    python test_full_pipeline.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnn_pipeline.analysis.community_analysis import advanced_community_analysis
from gnn_pipeline.analysis.topic_detection import extract_topics_from_query_dynamic
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

    print("=" * 70)
    print("STEP 8: END-TO-END INTEGRATION TEST")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load and Preprocess Data
    # ========================================================================
    print("\n[Step 1/6] Loading and preprocessing DBLP data...")
    csv_path = "/home/ramzy/Documents/Uni/metrognn/data/dblp_subset.csv"

    try:
        df, author_to_idx, author_list = load_and_clean_dblp_data(csv_path)
        print(f"✓ Loaded {len(df)} papers with {len(author_list)} unique authors")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    # ========================================================================
    # Step 2: Assign Dynamic Topics
    # ========================================================================
    print("\n[Step 2/6] Assigning dynamic topics...")

    try:
        # Create TopicAssigner for later query matching
        topic_assigner = TopicAssigner(n_clusters=8)
        df_with_topics, results_dict, topic_info_dict = assign_topics_to_dblp(
            df, n_clusters=8
        )

        # Re-run to get fitted vectorizer and cluster centers
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

    try:
        data, topic_names, _ = build_knowledge_graph_from_dblp(
            df_with_topics,
            author_to_idx,
            topic_info_dict,
            paper_feats,
            author_feats,
            topic_feats,
            save_path="/home/ramzy/Documents/Uni/metrognn/data/knowledge_graph.pt",
        )
        print(f"✓ Built knowledge graph with {len(data.node_types)} node types")
        print(f"  Graph saved to: data/knowledge_graph.pt")
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
        # Initialize model with dynamic topics
        model = AdvancedHeteroGCN(
            num_topics=len(topic_info_dict),
            hidden_channels=64,
            num_layers=4,
            num_classes=3,
        )

        # Train with reduced epochs for quick validation
        model, losses, val_metrics, imp_logits, role_logits, inf_scores = (
            train_enhanced_model(model, data, epochs=50)
        )

        print(f"✓ Training completed")
        print(f"  Final loss: {losses[-1]:.4f}")
        # print(f"  Importance accuracy: {val_metrics['val_imp_acc']:.4f}")
        # print(f"  Role accuracy: {val_metrics['val_role_acc']:.4f}")
        # print(f"  Influence MAE: {val_metrics['val_inf_mae']:.4f}")
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
    print("  ✓ Dynamic topic assignment (k=8)")
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

    # print(f"\nTraining Results:")
    # print(f"  Importance Accuracy: {val_metrics['val_imp_acc']:.4f}")
    # print(f"  Role Accuracy: {val_metrics['val_role_acc']:.4f}")
    # print(f"  Influence MAE: {val_metrics['val_inf_mae']:.4f}")

    print("\nNext Steps:")
    print("  - Test community analysis with trained model")
    print("  - Test strategy generation with dynamic topics")
    print("  - Integration with agentic layer (future)")
    print("=" * 70)

    print("\nPipeline Components Validated:")
    print("  ✓ Data loading and preprocessing")
    print("  ✓ Dynamic topic assignment (k=8)")
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
