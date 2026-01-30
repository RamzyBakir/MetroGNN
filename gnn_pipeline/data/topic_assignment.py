import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, List, Optional
import re


class TopicAssigner:
    """
    Assign papers to dynamic topics using TF-IDF + K-means clustering.
    Discovers topics from data rather than using predefined categories.
    """
    
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.kmeans = None
        self.feature_names = None
        
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing - optimized for titles and venue names.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or text is None or str(text).strip() == "":
            return ""
        
        text = str(text).lower()
        
        # Remove common venue metadata artifacts
        text = re.sub(r'early view|online version of record|published before inclusion', '', text, flags=re.IGNORECASE)
        
        # Keep only alphanumeric and spaces (remove punctuation)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_paper_text(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract text from title and venue (primary sources for topic classification).
        
        Args:
            df: DataFrame with 'title' and 'venue' columns
            
        Returns:
            Series of extracted text for each paper
        """
        texts = []
        for idx, row in df.iterrows():
            title = row.get('title', '')
            venue = row.get('venue', '')
            
            # Combine title and venue for stronger domain signal
            # Title provides content, venue provides domain/journal signal
            if title and str(title).strip():
                text = str(title)
                if venue and str(venue).strip():
                    text += " " + str(venue)
            else:
                # Fallback to venue only if title is missing
                text = str(venue) if venue else ""
            
            texts.append(text)
        
        return pd.Series(texts)
    
    def create_tfidf_vectors(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create TF-IDF vectors from text data - optimized for titles/venues.
        
        Args:
            texts: List of text documents
            
        Returns:
            tfidf_matrix: TF-IDF feature matrix
            feature_names: Array of feature names (words)
            feature_names_list: List of feature names
        """
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Increased for better coverage
            ngram_range=(1, 1),  # Unigrams only (less noise than bigrams for short text)
            min_df=1,  # Include terms appearing in at least 1 doc
            max_df=0.90,  # Ignore terms in more than 90% of docs (stricter filtering)
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        return tfidf_matrix.toarray(), self.feature_names
    
    def cluster_papers(self, tfidf_matrix: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Apply K-means clustering to TF-IDF vectors.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            
        Returns:
            cluster_labels: Array of cluster assignments for each paper
            silhouette_avg: Silhouette score (cluster quality metric)
            cluster_centers: Cluster centroids
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans.fit_predict(tfidf_matrix)
        cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate silhouette score (higher is better, range: -1 to 1)
        if self.n_clusters > 1:
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        return cluster_labels, silhouette_avg, cluster_centers
    
    def get_top_terms_per_cluster(self, cluster_centers: np.ndarray, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get top N terms for each cluster based on cluster centroids.
        
        Args:
            cluster_centers: Cluster centroids
            n_terms: Number of top terms to return
            
        Returns:
            Dictionary mapping cluster ID to list of top terms
        """
        top_terms_per_cluster = {}
        
        for cluster_id in range(self.n_clusters):
            # Get indices of top terms for this cluster
            top_indices = cluster_centers[cluster_id].argsort()[-n_terms:][::-1]
            top_terms = self.feature_names[top_indices].tolist()
            top_terms_per_cluster[cluster_id] = top_terms
        
        return top_terms_per_cluster
    
    def generate_topic_info(self, cluster_id: int, top_terms: List[str]) -> Dict:
        """
        Generate topic information including name and metadata.
        
        Args:
            cluster_id: ID of the cluster/topic
            top_terms: Top terms for this cluster
            
        Returns:
            Dictionary with topic name, id, and top terms
        """
        return {
            'topic_id': cluster_id,
            'name': f"Topic {cluster_id}",
            'top_terms': top_terms
        }
    
    def assign_topics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Main method: Assign dynamic topics to papers and return all relevant information.
        
        Args:
            df: DataFrame with papers (must have 'title' and 'venue' columns)
            
        Returns:
            df_with_topics: DataFrame with added 'topic_id' column
            results_dict: Dictionary with clustering results and analysis
            topic_info_dict: Dictionary mapping topic ID to topic information
        """
        # Step 1: Extract text
        texts = self.extract_paper_text(df)
        texts_preprocessed = [self.preprocess_text(text) for text in texts]
        
        # Step 2: Create TF-IDF vectors
        tfidf_matrix, feature_names = self.create_tfidf_vectors(texts_preprocessed)
        
        # Step 3: Cluster papers
        cluster_labels, silhouette_avg, cluster_centers = self.cluster_papers(tfidf_matrix)
        
        # Step 4: Get top terms per cluster
        top_terms_per_cluster = self.get_top_terms_per_cluster(cluster_centers)
        
        # Step 5: Generate topic information (dynamic topic names)
        topic_info_dict = {}
        for cluster_id, top_terms in top_terms_per_cluster.items():
            topic_info_dict[cluster_id] = self.generate_topic_info(cluster_id, top_terms)
        
        # Step 6: Create topic names dictionary (for backward compatibility)
        topic_names = {tid: info['name'] for tid, info in topic_info_dict.items()}
        
        # Step 7: Cluster IDs ARE topic IDs (no mapping needed)
        topic_labels = cluster_labels  # Direct mapping
        
        # Add to dataframe
        df_with_topics = df.copy()
        df_with_topics['topic_id'] = topic_labels
        df_with_topics['topic_name'] = df_with_topics['topic_id'].map(topic_names)
        
        # Store TF-IDF vectorizer for query matching later
        self.tfidf_vectors = tfidf_matrix
        self.cluster_centers_ = cluster_centers
        
        # Prepare results
        results_dict = {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'top_terms_per_cluster': top_terms_per_cluster,
            'topic_distribution': df_with_topics['topic_id'].value_counts().to_dict(),
            'topic_info': topic_info_dict
        }
        
        return df_with_topics, results_dict, topic_info_dict
    
    def get_vectorizer(self) -> Optional[TfidfVectorizer]:
        """
        Get the fitted TF-IDF vectorizer for query matching.
        
        Returns:
            Fitted TfidfVectorizer or None if not yet fitted
        """
        return self.vectorizer
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers for query matching.
        
        Returns:
            Cluster centers or None if not yet computed
        """
        return getattr(self, 'cluster_centers_', None)


def assign_topics_to_dblp(df: pd.DataFrame, n_clusters: int = 8) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Convenience function to assign dynamic topics to DBLP dataset.
    
    Args:
        df: DataFrame with DBLP papers (must have 'title' and 'venue' columns)
        n_clusters: Number of topics to discover (default: 8)
        
    Returns:
        df_with_topics: DataFrame with added topic assignments
        results_dict: Clustering results and analysis
        topic_info_dict: Dictionary mapping topic ID to topic information
    """
    assigner = TopicAssigner(n_clusters=n_clusters)
    df_with_topics, results_dict, topic_info_dict = assigner.assign_topics(df)
    
    return df_with_topics, results_dict, topic_info_dict


if __name__ == "__main__":
    # This is for testing only - user should run this separately
    print("This module provides dynamic topic assignment functionality.")
    print("Use assign_topics_to_dblp() function with your DBLP DataFrame.")
    print("Topics are discovered from data, not predefined.")