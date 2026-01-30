def extract_topics_from_query(query, topic_names, include_recency=True):

    query_lower = query.lower()


    topic_hierarchy = {
        0: {"ai": 2, "artificial intelligence": 3, "neural networks": 2, "expert systems": 1},
        1: {"machine learning": 3, "ml": 2, "supervised learning": 2, "unsupervised learning": 2,
            "reinforcement learning": 2, "predictive modeling": 1},
        2: {"deep learning": 3, "neural networks": 2, "cnn": 2, "rnn": 2, "transformer": 2, "attention": 1},
        3: {"natural language processing": 3, "nlp": 2, "language models": 2, "text mining": 2,
            "sentiment analysis": 1, "chatbots": 1, "translation": 1},
        4: {"computer vision": 3, "cv": 2, "image processing": 2, "object detection": 1, "facial recognition": 1},
        5: {"data mining": 3, "knowledge discovery": 2, "pattern recognition": 2, "big data": 1},
        6: {"cloud computing": 3, "distributed systems": 2, "aws": 1, "azure": 1, "serverless": 1},
        7: {"cybersecurity": 3, "security": 2, "privacy": 2, "encryption": 1, "malware": 1, "threat detection": 1}
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
    relevant_topics = [topic[0] for topic in sorted_topics if topic[1] >= confidence_threshold]


    if not relevant_topics:
        if any(word in query_lower for word in ["learn", "study", "get into", "beginner"]):

            if any(word in query_lower for word in ["ai", "artificial", "machine"]):
                relevant_topics = [0, 1]
            elif any(word in query_lower for word in ["security", "cyber", "privacy"]):
                relevant_topics = [7]
            elif any(word in query_lower for word in ["language", "text", "nlp"]):
                relevant_topics = [3]
        elif any(word in query_lower for word in ["research", "phd", "academic", "contribute"]):

            relevant_topics = [1, 2, 3]
        else:

            relevant_topics = [0, 1]

    return relevant_topics[:2]


def extract_topics_from_query_dynamic(query: str, topic_assigner, topic_info_dict: dict) -> list:
    """
    Match user query to dynamic topics using TF-IDF similarity.
    
    Args:
        query: User query string
        topic_assigner: TopicAssigner instance with fitted vectorizer and cluster centers
        topic_info_dict: Topic information dictionary
        
    Returns:
        List of relevant topic IDs (up to 2)
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return list(topic_info_dict.keys())[:2]
    
    # Preprocess query
    query_clean = topic_assigner.preprocess_text(query)
    
    if not query_clean or not query_clean.strip():
        return list(topic_info_dict.keys())[:2]
    
    # Get vectorizer and cluster centers
    vectorizer = topic_assigner.get_vectorizer()
    cluster_centers = topic_assigner.get_cluster_centers()
    
    if vectorizer is None or cluster_centers is None:
        return list(topic_info_dict.keys())[:2]
    
    try:
        # Vectorize query using fitted TF-IDF
        query_vector = vectorizer.transform([query_clean]).toarray()[0]
        
        # Compute cosine similarity to each cluster center
        similarities = cosine_similarity([query_vector], cluster_centers)[0]
        
        # Get top 2 topics
        top_indices = similarities.argsort()[-2:][::-1]
        
        return top_indices.tolist()
    except Exception as e:
        return list(topic_info_dict.keys())[:2]