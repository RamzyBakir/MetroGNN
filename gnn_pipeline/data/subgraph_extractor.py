"""
Extract 1-hop subgraphs from the full heterogeneous knowledge graph
given a set of seed paper IDs.

All operations are vectorized with PyTorch — no Python loops over edges.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Set, Tuple


class SubgraphExtractor:
    """
    Extracts a 1-hop subgraph from the full knowledge graph
    given a set of seed paper node IDs.
    """

    def __init__(self, full_graph: HeteroData):
        self.full_graph = full_graph

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _filter_and_remap_edges(
        edge_index: torch.Tensor,
        src_map: Dict[int, int],
        dst_map: Dict[int, int],
    ) -> torch.Tensor:
        """
        Keep only edges whose both endpoints are in the maps,
        and remap IDs to the subgraph's local numbering.
        Returns a (2, E') tensor or empty (2, 0) tensor.
        """
        if edge_index.shape[1] == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        src_set = torch.tensor(sorted(src_map.keys()), dtype=torch.long)
        dst_set = torch.tensor(sorted(dst_map.keys()), dtype=torch.long)

        src_in = torch.isin(edge_index[0], src_set)
        dst_in = torch.isin(edge_index[1], dst_set)
        mask = src_in & dst_in

        if not mask.any():
            return torch.zeros((2, 0), dtype=torch.long)

        filtered = edge_index[:, mask]

        # Remap using vectorised lookup tables
        max_src = max(src_map.keys()) + 1
        max_dst = max(dst_map.keys()) + 1

        src_lut = torch.full((max_src,), -1, dtype=torch.long)
        for orig, new in src_map.items():
            src_lut[orig] = new

        dst_lut = torch.full((max_dst,), -1, dtype=torch.long)
        for orig, new in dst_map.items():
            dst_lut[orig] = new

        remapped = torch.stack([src_lut[filtered[0]], dst_lut[filtered[1]]])
        return remapped

    # ── main entry point ────────────────────────────────────────────────

    def extract_1hop_subgraph(
        self,
        retrieved_paper_ids: Set[int],
    ) -> HeteroData:
        """
        Extract the 1-hop neighbourhood of *retrieved_paper_ids*:
          papers  →  their authors  →  their topics
                                    →  author-author collaborations

        Returns a new HeteroData with remapped node/edge indices and a
        `._id_maps` dict for translating back to the full graph.
        """
        fg = self.full_graph
        if "paper" not in fg:
            return HeteroData(), {}

        n_papers = fg["paper"].num_nodes
        paper_ids_valid = sorted(pid for pid in retrieved_paper_ids if pid < n_papers)
        if not paper_ids_valid:
            return HeteroData(), {}

        paper_tensor = torch.tensor(paper_ids_valid, dtype=torch.long)

        # ── 1. Discover connected authors ───────────────────────────────
        auth_edges = fg["author", "authored", "paper"].edge_index  # (2, E)
        # papers appear in row 1
        paper_mask = torch.isin(auth_edges[1], paper_tensor)
        author_ids = auth_edges[0, paper_mask].unique().tolist()

        # ── 2. Discover connected topics ────────────────────────────────
        pt_edges = fg["paper", "has_topic", "topic"].edge_index
        pt_mask = torch.isin(pt_edges[0], paper_tensor)
        topic_ids = pt_edges[1, pt_mask].unique().tolist()

        # ── 3. Build ID maps  (original → subgraph) ────────────────────
        paper_id_map = {orig: new for new, orig in enumerate(paper_ids_valid)}
        author_id_map = {orig: new for new, orig in enumerate(author_ids)}
        topic_id_map = {orig: new for new, orig in enumerate(topic_ids)}

        # Reverse maps (subgraph → original)
        paper_rev = {v: k for k, v in paper_id_map.items()}
        author_rev = {v: k for k, v in author_id_map.items()}
        topic_rev = {v: k for k, v in topic_id_map.items()}

        # ── 4. Create subgraph with node features ──────────────────────
        subgraph = HeteroData()

        if paper_ids_valid:
            subgraph["paper"].x = fg["paper"].x[paper_ids_valid]

        if author_ids:
            subgraph["author"].x = fg["author"].x[author_ids]

        if topic_ids and "topic" in fg:
            subgraph["topic"].x = fg["topic"].x[topic_ids]

        # ── 5. Extract & remap edges ───────────────────────────────────

        edge_specs = [
            (("author", "authored", "paper"), author_id_map, paper_id_map),
            (("paper", "authored_by", "author"), paper_id_map, author_id_map),
            (("paper", "has_topic", "topic"), paper_id_map, topic_id_map),
            (("topic", "has_paper", "paper"), topic_id_map, paper_id_map),
            (("author", "collaborates_with", "author"), author_id_map, author_id_map),
        ]

        for edge_type, src_map, dst_map in edge_specs:
            if edge_type in fg.edge_types and src_map and dst_map:
                ei = fg[edge_type].edge_index
                remapped = self._filter_and_remap_edges(ei, src_map, dst_map)
                if remapped.shape[1] > 0:
                    subgraph[edge_type].edge_index = remapped

        # ── 6. Store reverse maps for later ────────────────────────────
        id_maps = {
            "paper_id_map": paper_rev,
            "author_id_map": author_rev,
            "topic_id_map": topic_rev,
            "original_paper_ids": paper_ids_valid,
            "original_author_ids": author_ids,
            "original_topic_ids": topic_ids,
        }

        return subgraph, id_maps
