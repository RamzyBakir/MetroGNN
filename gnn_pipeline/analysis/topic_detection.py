def calculate_topic_relevance_score(
    query: str, topic_info_dict: dict, detected_topic_ids: list
) -> dict:
    """
    Calculate relevance scores for each detected topic based on query terms.

    Args:
        query: User query
        topic_info_dict: Topic information
        detected_topic_ids: Detected topic IDs

    Returns:
        Dictionary mapping topic_id to relevance score
    """
    query_terms = set(query.lower().split())
    relevance_scores = {}

    for tid in detected_topic_ids:
        info = topic_info_dict.get(tid, topic_info_dict.get(str(tid)))
        if info is None:
            relevance_scores[int(tid)] = 0.0
            continue

        top_terms = info.get("top_terms", [])
        match_score = 0.0

        # Count exact and partial matches
        for qterm in query_terms:
            for tterm in top_terms:
                # Exact match: higher score
                if qterm == tterm.lower():
                    match_score += 2.0
                # Partial match: lower score
                elif qterm in tterm.lower() or tterm.lower() in qterm:
                    match_score += 1.0

        relevance_scores[int(tid)] = match_score

    return relevance_scores


def extract_topics_from_query(query, topic_names, include_recency=True):

    query_lower = query.lower()

    topic_hierarchy = {
        0: {
            "ai": 2,
            "artificial intelligence": 3,
            "neural networks": 2,
            "expert systems": 1,
        },
        1: {
            "machine learning": 3,
            "ml": 2,
            "supervised learning": 2,
            "unsupervised learning": 2,
            "reinforcement learning": 2,
            "predictive modeling": 1,
        },
        2: {
            "deep learning": 3,
            "neural networks": 2,
            "cnn": 2,
            "rnn": 2,
            "transformer": 2,
            "attention": 1,
        },
        3: {
            "natural language processing": 3,
            "nlp": 2,
            "language models": 2,
            "text mining": 2,
            "sentiment analysis": 1,
            "chatbots": 1,
            "translation": 1,
        },
        4: {
            "computer vision": 3,
            "cv": 2,
            "image processing": 2,
            "object detection": 1,
            "facial recognition": 1,
        },
        5: {
            "data mining": 3,
            "knowledge discovery": 2,
            "pattern recognition": 2,
            "big data": 1,
        },
        6: {
            "cloud computing": 3,
            "distributed systems": 2,
            "aws": 1,
            "azure": 1,
            "serverless": 1,
        },
        7: {
            "cybersecurity": 3,
            "security": 2,
            "privacy": 2,
            "encryption": 1,
            "malware": 1,
            "threat detection": 1,
        },
    }

    topic_scores = {}
    for topic_id, keywords in topic_hierarchy.items():
        score = 0
        for keyword, weight in keywords.items():
            if keyword in query_lower:
                score += weight
        if score > 0:
            topic_scores[topic_id] = score

    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    confidence_threshold = 2.0
    relevant_topics = [
        topic[0] for topic in sorted_topics if topic[1] >= confidence_threshold
    ]

    if not relevant_topics:
        if any(
            word in query_lower for word in ["learn", "study", "get into", "beginner"]
        ):
            if any(word in query_lower for word in ["ai", "artificial", "machine"]):
                relevant_topics = [0, 1]
            elif any(word in query_lower for word in ["security", "cyber", "privacy"]):
                relevant_topics = [7]
            elif any(word in query_lower for word in ["language", "text", "nlp"]):
                relevant_topics = [3]
        elif any(
            word in query_lower
            for word in ["research", "phd", "academic", "contribute"]
        ):
            relevant_topics = [1, 2, 3]
        else:
            relevant_topics = [0, 1]

    return relevant_topics[:2]


def extract_topics_from_query_dynamic(
    query: str, topic_assigner, topic_info_dict: dict
) -> list:
    """Match a query to persisted MiniLM topic centroids."""
    try:
        return [topic_id for topic_id, _ in topic_assigner.rank_query_topics(query, top_k=2)]
    except (AttributeError, RuntimeError, ValueError):
        return fallback_keyword_match(query, topic_info_dict)[:2]


def fallback_keyword_match(
    query: str, topic_info_dict: dict, n_results: int = 2
) -> list:
    """
    Fallback keyword matching when TF-IDF fails.
    """
    query_terms = set(query.lower().split())
    scores = []

    for topic_id, info in topic_info_dict.items():
        top_terms = info.get("top_terms", [])
        match_count = sum(
            1 for term in top_terms if any(qt in term.lower() for qt in query_terms)
        )
        scores.append((int(topic_id), match_count))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in scores[:n_results]]
