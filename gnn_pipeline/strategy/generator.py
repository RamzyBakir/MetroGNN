import numpy as np

class StrategyGenerator:


    def __init__(self, topic_names=None):
        self.topic_names = topic_names or self.get_default_topic_names()

    def get_default_topic_names(self):

        return {
            0: "Artificial Intelligence", 1: "Machine Learning", 2: "Deep Learning",
            3: "Natural Language Processing", 4: "Computer Vision",
            5: "Data Mining", 6: "Cloud Computing", 7: "Cybersecurity"
        }

    def assemble_strategy_context(self, community_data, user_query):

        user_intent = self.classify_user_intent(user_query)


        enhanced_analysis = community_data.get('enhanced_analysis', {})
        if not enhanced_analysis:
            enhanced_analysis = {
                'research_quality': 0.7,
                'collaboration_potential': 0.6,
                'topic_alignment': 0.8,
                'research_freshness': 0.5,
                'topic_specialization': 0.6,
                'research_productivity': 0.7,
                'influence_concentration': 0.4
            }


        strategy_context = {
            'community_profile': {
                'name': community_data.get('community_name', 'Research Community'),
                'size': community_data.get('community_metrics', {}).get('size', 0),
                'quality_scores': enhanced_analysis,
                'key_strengths': community_data.get('key_strengths', []),
                'weaknesses': community_data.get('weaknesses', [])
            },

            'user_profile': {
                'query': user_query,
                'detected_topics': community_data.get('query_topics', []),
                'intent': user_intent,
                'experience_level': self.infer_experience_level(user_query),
                'topic_names': self.topic_names
            },

            'key_members': self.categorize_key_members(community_data.get('key_members', []))
        }

        return strategy_context

    def classify_user_intent(self, query):

        query_lower = query.lower()

        if any(word in query_lower for word in ["learn", "study", "beginner", "new to"]):
            return "learning"
        elif any(word in query_lower for word in ["research", "phd", "academic", "publish"]):
            return "research"
        elif any(word in query_lower for word in ["career", "job", "industry", "professional"]):
            return "career"
        else:
            return "general"

    def infer_experience_level(self, query):

        query_lower = query.lower()

        if any(word in query_lower for word in ["beginner", "new", "start", "learn"]):
            return "beginner"
        elif any(word in query_lower for word in ["advanced", "expert", "deep"]):
            return "advanced"
        else:
            return "intermediate"

    def categorize_key_members(self, key_members):

        categorized = {'leaders': [], 'bridges': [], 'experts': [], 'influencers': []}

        for member in key_members:

            if "Leader" in member.get('role', ''):
                categorized['leaders'].append(member)


            if member.get('betweenness', 0) > 0.1:
                categorized['bridges'].append(member)


            if member.get('topic_specialization', 0) > 0.7:
                categorized['experts'].append(member)


            if member.get('global_influence', 0) > 0.8:
                categorized['influencers'].append(member)

        return categorized

    def generate_comprehensive_strategy(self, community_data, user_query):

        strategy_context = self.assemble_strategy_context(community_data, user_query)

        return {
            'metadata': {
                'query': user_query,
                'community': strategy_context['community_profile']['name'],
                'user_intent': strategy_context['user_profile']['intent']
            },
            'strategy_context': strategy_context,
            'simple_strategy': f"Learning path for {user_query}"
        }