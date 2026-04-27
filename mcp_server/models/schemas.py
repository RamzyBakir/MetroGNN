from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Shared ───────────────────────────────────────────────────────────────


class AuthorProfile(BaseModel):
    author_id: int = Field(description="Author node ID in the knowledge graph")
    importance_class: int = Field(
        description="Importance class index (0=Low, 1=Medium, 2=High)"
    )
    importance_label: str = Field(description="Human-readable importance label")
    role_class: int = Field(
        description="Role class index (0=Hub, 1=Active, 2=Occasional, 3=Isolated)"
    )
    role_label: str = Field(description="Human-readable role label")
    influence: float = Field(description="GNN-predicted influence score")
    paper_count: int = Field(description="Number of papers authored")
    collaboration_count: int = Field(description="Number of collaborators")
    topic_diversity: float = Field(
        description="Spread of topics the author publishes in"
    )
    years_active: float = Field(
        description="Number of years the author has been active"
    )
    citation_impact: float = Field(description="Aggregate citation impact")


class PaperResult(BaseModel):
    paper_id: int = Field(description="Paper node ID (= DataFrame row index)")
    title: str = Field(description="Paper title")
    year: Optional[int] = Field(default=None, description="Publication year")
    citations: int = Field(default=0, description="Citation count")
    venue: str = Field(default="", description="Publication venue")
    topic_id: Optional[int] = Field(default=None, description="Assigned topic ID")
    topic_name: str = Field(default="", description="Topic name")
    authors: List[AuthorProfile] = Field(
        default_factory=list, description="Author profiles"
    )
    aggregated_influence: float = Field(
        default=0.0, description="Mean GNN influence of authors"
    )
    semantic_similarity: float = Field(
        default=0.0, description="Cosine similarity to query"
    )
    relevance_score: float = Field(
        default=0.0, description="Combined 0.6*semantic + 0.4*influence"
    )
    num_authors: int = Field(default=0, description="Number of authors")


# ── Training ─────────────────────────────────────────────────────────────


class TrainPipelineInput(BaseModel):
    epochs: int = Field(default=300, ge=1, le=2000, description="Training epochs")
    hidden_channels: int = Field(default=64, description="Hidden layer size")
    out_channels: int = Field(default=32, description="Output embedding size")
    num_layers: int = Field(default=4, ge=1, le=8, description="Number of GNN layers")
    n_clusters: int = Field(
        default=8, ge=2, le=20, description="Number of topic clusters"
    )
    force_retrain: bool = Field(
        default=False, description="Retrain even if model already exists"
    )


class TrainPipelineOutput(BaseModel):
    status: Literal["completed", "failed"] = Field(description="Final status")
    epochs_trained: int = Field(description="Actual epochs run (may stop early)")
    final_train_loss: float = Field(description="Final training loss")
    model_path: str = Field(description="Where model weights were saved")
    graph_path: str = Field(description="Where the knowledge graph was saved")
    graph_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Node/edge counts"
    )


class BuildVectorStoreInput(BaseModel):
    force_reingest: bool = Field(
        default=False, description="Re-ingest even if store exists"
    )


class BuildVectorStoreOutput(BaseModel):
    status: Literal["completed", "skipped", "failed"] = Field(
        description="Final status"
    )
    papers_indexed: int = Field(description="Number of papers in the vector store")
    persist_dir: str = Field(description="Path to the ChromaDB directory")


# ── Inference ────────────────────────────────────────────────────────────


class QueryPapersInput(BaseModel):
    query: str = Field(description="Free-text search query")
    top_k: int = Field(
        default=50, ge=1, le=500, description="Number of papers to retrieve"
    )
    interests: Optional[List[str]] = Field(
        default=None, description="Extra interest terms to boost relevance"
    )


class QueryPapersOutput(BaseModel):
    query: str = Field(description="The search query used")
    total_results: int = Field(description="Number of papers returned")
    papers: List[Dict[str, Any]] = Field(
        description="Retrieved papers with metadata and distances"
    )


class RunInferenceInput(BaseModel):
    query: str = Field(description="Free-text research query")
    experience_level: Literal["beginner", "intermediate", "advanced"] = Field(
        default="intermediate", description="User experience level"
    )
    interests: Optional[List[str]] = Field(
        default=None, description="Interest keywords"
    )
    goals: Literal["learn", "research", "career"] = Field(
        default="research", description="User goal"
    )
    top_papers: int = Field(
        default=20, ge=1, le=100, description="Final papers to return"
    )
    retrieval_k: int = Field(
        default=100, ge=1, le=500, description="Top-K for vector retrieval"
    )
    sort_by: Literal["influence", "citations", "year"] = Field(
        default="influence", description="Sort criterion"
    )


class InferenceSummary(BaseModel):
    total_papers: int = Field(description="Total papers in dataset")
    retrieved_papers: int = Field(description="Papers retrieved from vector store")
    relevant_papers: int = Field(description="Papers returned after filtering")
    top_influential_authors: List[AuthorProfile] = Field(
        default_factory=list, description="Top 10 authors by influence"
    )


class RunInferenceOutput(BaseModel):
    user_profile: Dict[str, Any] = Field(description="Query metadata")
    papers: List[PaperResult] = Field(description="Ranked paper results")
    summary: InferenceSummary = Field(description="Aggregated summary")


class GetAuthorProfilesInput(BaseModel):
    author_ids: Optional[List[int]] = Field(
        default=None,
        description="Specific author IDs to look up. Omit for all authors.",
    )
    importance_filter: Optional[Literal["Low", "Medium", "High"]] = Field(
        default=None, description="Filter by importance label"
    )
    role_filter: Optional[Literal["Hub", "Active", "Occasional", "Isolated"]] = Field(
        default=None, description="Filter by role label"
    )
    min_influence: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum influence"
    )
    limit: int = Field(default=50, ge=1, le=500, description="Max authors to return")


class GetAuthorProfilesOutput(BaseModel):
    total_authors: int = Field(description="Total authors in the graph")
    returned_count: int = Field(description="Number of authors in this response")
    authors: List[AuthorProfile] = Field(description="Author profiles")


# ── Analysis ─────────────────────────────────────────────────────────────


class AnalyzeCommunityInput(BaseModel):
    query: str = Field(description="User research query to detect topics from")
    topic_ids: Optional[List[int]] = Field(
        default=None,
        description="Explicit topic IDs to analyze. Auto-detected if omitted.",
    )


class ResearcherData(BaseModel):
    id: str = Field(description="Researcher node ID (e.g. R42)")
    global_influence: float = Field(description="GNN influence score")
    centrality: float = Field(description="Degree centrality in community subgraph")
    betweenness: float = Field(description="Betweenness centrality")
    clustering: float = Field(description="Local clustering coefficient")
    topic_specialization: float = Field(
        description="Fraction of papers in query topics"
    )
    role: str = Field(description="Predicted or heuristic role")
    importance: str = Field(description="Predicted importance label")
    composite_score: float = Field(description="Weighted composite score")


class CommunityMetrics(BaseModel):
    size: int = Field(description="Number of community members")
    cohesion: float = Field(description="Network density")
    avg_clustering: float = Field(description="Mean clustering coefficient")
    internal_edges: int = Field(description="Collaboration edges within community")
    research_productivity: float = Field(description="Avg papers per researcher")
    collaboration_potential: float = Field(description="Edge density")
    influence_concentration: float = Field(description="Top-3 influence share")


class AnalyzeCommunityOutput(BaseModel):
    community_name: str = Field(description="Detected topic names joined")
    query_topics: List[int] = Field(description="Topic IDs used for analysis")
    key_members: List[ResearcherData] = Field(description="High-influence members")
    other_members: List[ResearcherData] = Field(
        description="Remaining community members"
    )
    community_metrics: CommunityMetrics = Field(description="Aggregate network metrics")
    enhanced_analysis: Dict[str, Any] = Field(description="Extended quality metrics")


class DetectTopicsInput(BaseModel):
    query: str = Field(description="Free-text query")
    method: Literal["dynamic", "keyword"] = Field(
        default="dynamic",
        description="Detection method: TF-IDF cosine or keyword matching",
    )


class TopicInfo(BaseModel):
    topic_id: int = Field(description="Topic cluster ID")
    name: str = Field(default="", description="Topic name if available")
    relevance_score: float = Field(default=0.0, description="Relevance to the query")


class DetectTopicsOutput(BaseModel):
    query: str = Field(description="The input query")
    detected_topics: List[TopicInfo] = Field(description="Matched topics with scores")
    all_topic_names: Dict[int, str] = Field(
        default_factory=dict, description="Full topic ID -> name mapping"
    )


class GenerateStrategyInput(BaseModel):
    query: str = Field(description="User research query")
    topic_ids: Optional[List[int]] = Field(
        default=None, description="Explicit topic IDs. Auto-detected if omitted."
    )


class GenerateStrategyOutput(BaseModel):
    metadata: Dict[str, Any] = Field(description="Strategy metadata")
    executive_summary: str = Field(description="High-level plan summary")
    community_analysis: Dict[str, Any] = Field(description="Community insights")
    strategic_plan: Dict[str, Any] = Field(
        description="Learning path, skills, projects, timeline"
    )
    implementation_guide: Dict[str, Any] = Field(description="Actionable next steps")
    success_framework: Dict[str, Any] = Field(
        description="Tracking and adaptation plan"
    )


# ── Utility ──────────────────────────────────────────────────────────────


class GraphInfoOutput(BaseModel):
    node_types: Dict[str, int] = Field(description="Node type -> count")
    edge_types: Dict[str, int] = Field(description="Edge type (src->dst) -> edge count")
    feature_shapes: Dict[str, List[int]] = Field(
        description="Node type -> feature tensor shape"
    )
    graph_path: str = Field(description="Path the graph was loaded from")
