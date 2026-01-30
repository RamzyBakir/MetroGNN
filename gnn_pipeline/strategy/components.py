from datetime import datetime
import numpy as np

class StrategyQualityValidator:


    def validate_strategy_quality(self, final_plan):

        checks = {
            "completeness": self.check_completeness(final_plan),
            "actionability": self.check_actionability(final_plan),
            "personalization": self.check_personalization(final_plan),
            "realism": self.check_realism(final_plan)
        }

        quality_score = sum(checks.values()) / len(checks)

        if quality_score < 0.7:
            return self.enhance_strategy_quality(final_plan, checks)
        return final_plan

    def check_completeness(self, plan):

        strategic_plan = plan.get('strategic_plan', {})


        if isinstance(strategic_plan, str):
            return False

        essential_sections = ['learning_path', 'collaboration_strategy', 'success_metrics']
        return all(section in strategic_plan for section in essential_sections)

    def check_actionability(self, plan):

        strategy_text = str(plan)
        actionable_indicators = ['month', 'week', 'specific', 'connect with', 'learn']
        return sum(1 for indicator in actionable_indicators if indicator in strategy_text.lower()) / len(
            actionable_indicators)

    def check_personalization(self, plan):

        personalization_indicators = ['based on', 'tailored', 'specific to', 'considering']
        strategy_text = str(plan)
        return sum(1 for indicator in personalization_indicators if indicator in strategy_text.lower()) / len(
            personalization_indicators)

    def check_realism(self, plan):

        strategic_plan = plan.get('strategic_plan', {})


        if isinstance(strategic_plan, str):
            timeline_text = strategic_plan
        else:
            timeline_text = strategic_plan.get('timeline', '')


        if '12' in timeline_text or 'year' in timeline_text:
            return 0.8
        elif '6' in timeline_text or 'months' in timeline_text:
            return 0.6
        return 0.4

    def enhance_strategy_quality(self, plan, failed_checks):

        enhanced_plan = plan.copy()

        if not failed_checks['completeness']:
            enhanced_plan = self.add_missing_sections(enhanced_plan)

        if not failed_checks['actionability']:
            enhanced_plan = self.make_recommendations_actionable(enhanced_plan)

        return enhanced_plan

    def add_missing_sections(self, plan):

        strategic_plan = plan.get('strategic_plan', {})

        if 'success_metrics' not in strategic_plan:
            strategic_plan['success_metrics'] = [
                "Complete 3 portfolio projects",
                "Establish professional network",
                "Master core domain concepts"
            ]

        if 'timeline' not in strategic_plan:
            strategic_plan['timeline'] = "6-12 month progressive learning path"

        plan['strategic_plan'] = strategic_plan
        return plan

    def make_recommendations_actionable(self, plan):

        strategy_text = str(plan)
        if 'learn' in strategy_text.lower() and 'specific' not in strategy_text.lower():
            plan['strategic_plan']['learning_path'] += " Focus on specific, measurable skills each month."

        return plan

class DynamicContentGenerator:


    def generate_all_components(self, strategy_context, topic_info_dict=None):
        
        return {
            'learning_path': self.generate_learning_path(strategy_context),
            'collaboration_strategy': self.generate_collaboration_strategy(strategy_context),
            'skill_development': self.generate_skill_development(strategy_context, topic_info_dict),
            'research_projects': self.generate_research_projects(strategy_context, topic_info_dict),
            'timeline': self.generate_timeline(strategy_context),
            'success_metrics': self.generate_success_metrics(strategy_context)
        }

    def generate_learning_path(self, context):

        community_size = context['community_profile']['size']
        research_quality = context['community_profile']['quality_scores']['research_quality']
        user_intent = context['user_profile']['intent']

        if user_intent == 'learning':
            if community_size > 10 and research_quality > 0.8:
                return "Comprehensive 12-month learning path: Foundation (1-3mo) → Core skills (4-6mo) → Advanced topics (7-9mo) → Specialization (10-12mo)"
            elif community_size < 5:
                return "Focused 8-month program: Intensive fundamentals (1-4mo) → Applied projects (5-8mo)"
            else:
                return "Balanced 10-month curriculum: Theory + practice with gradual complexity increase"

        elif user_intent == 'research':
            return "Research-intensive program emphasizing methodology, literature review, and publication"

        else:
            return "Industry-focused development with practical projects and professional networking"

    def generate_collaboration_strategy(self, context):

        strategies = []
        key_members = context['key_members']
        collaboration_potential = context['community_profile']['quality_scores']['collaboration_potential']


        for member_type in ['leaders', 'bridges', 'experts']:
            for member in key_members.get(member_type, [])[:2]:
                if member_type == 'leaders':
                    strategies.append(f"Connect with {member['id']} for research direction and mentorship")
                elif member_type == 'bridges':
                    strategies.append(f"Collaborate with {member['id']} for cross-community opportunities")
                elif member_type == 'experts':
                    strategies.append(f"Learn specialized techniques from {member['id']}")


        if collaboration_potential < 0.3:
            strategies.append("Focus on deep relationships with few experts due to low collaboration density")
        elif collaboration_potential > 0.6:
            strategies.append("Engage with multiple research groups in this highly collaborative community")

        return strategies

    def generate_skill_development(self, context, topic_info_dict=None):
        
        skills = {
            'foundational': ["critical thinking", "research methodology", "domain fundamentals"],
            'technical': self.get_domain_technical_skills(context, topic_info_dict),
            'professional': ["academic writing", "presentation skills", "collaboration"]
        }
        

        if context['community_profile']['quality_scores']['research_quality'] > 0.8:
            skills['technical'].extend(["advanced research techniques", "methodological rigor"])
        
        if context['community_profile']['quality_scores']['collaboration_potential'] > 0.5:
            skills['professional'].extend(["team research", "interdisciplinary communication"])
        
        return skills

    def get_domain_technical_skills(self, context, topic_info_dict=None):
        
        primary_topic = context['user_profile']['detected_topics'][0] if context['user_profile'][
            'detected_topics'] else 0
        
        # Use topic info if available, otherwise fallback to hardcoded
        if topic_info_dict and primary_topic in topic_info_dict:
            topic_terms = topic_info_dict[primary_topic]['top_terms']
        else:
            topic_name = context['user_profile'].get('topic_names', {}).get(primary_topic, "AI").lower()
            topic_terms = [topic_name]
        
        # Generic skill mapping based on top terms
        skill_map = {
            'programming': ['Python programming', 'TensorFlow', 'PyTorch'],
            'algorithm': ['Data Structures', 'Algorithms', 'Optimization'],
            'graph': ['Graph Theory', 'Network Analysis', 'Graph Databases'],
            'analysis': ['Statistical Analysis', 'Data Visualization', 'Experimental Design'],
            'learning': ['Machine Learning', 'Deep Learning', 'Neural Networks'],
            'system': ['Distributed Systems', 'Cloud Computing', 'System Design'],
            'wireless': ['Wireless Networks', 'IoT', 'Communication Protocols'],
            'optimization': ['Linear Programming', 'Convex Optimization', 'Combinatorial Optimization'],
            'control': ['Control Theory', 'Feedback Systems', 'System Dynamics'],
            'data': ['Data Mining', 'Data Wrangling', 'Data Engineering'],
            'network': ['Network Protocols', 'TCP/IP', 'Network Security']
        }
        
        skills = []
        for term, related_skills in skill_map.items():
            if any(term in t.lower() for t in topic_terms):
                skills.extend(related_skills)
        
        # Remove duplicates and limit to reasonable number
        skills = list(set(skills))[:12]
        
        if not skills:
            skills = ["data analysis", "research design", "experimental methodology"]
        
        return skills
    
    def generate_research_projects(self, context, topic_info_dict=None):
        
        projects = ["literature review and gap analysis", "methodology implementation", "empirical validation"]

        primary_topic = context['user_profile']['detected_topics'][0] if context['user_profile'][
            'detected_topics'] else 0
        
        # Use topic info if available
        if topic_info_dict and primary_topic in topic_info_dict:
            topic_terms = topic_info_dict[primary_topic]['top_terms'][:8]
            for term in topic_terms:
                projects.append(f"Research project focusing on {term} and its applications")
        else:
            # Fallback to hardcoded approach
            topic_name = context['user_profile'].get('topic_names', {}).get(primary_topic, "AI").lower()
            if any(word in topic_name for word in ["machine", "learning"]):
                projects.extend(["model comparison study", "hyperparameter optimization", "real-world deployment"])
            elif any(word in topic_name for word in ["security"]):
                projects.extend(["security audit framework", "threat modeling", "defense mechanism evaluation"])
        
        return projects

    def generate_timeline(self, context):

        base_timeline = "6-12 months"


        community_size = context['community_profile']['size']
        research_quality = context['community_profile']['quality_scores']['research_quality']

        if community_size > 10 and research_quality > 0.8:
            return "9-12 month comprehensive program"
        elif community_size < 5:
            return "6-8 month focused intensive"
        else:
            return base_timeline

    def generate_success_metrics(self, context):

        metrics = [
            "Complete portfolio of practical projects",
            "Establish professional network within the community",
            "Demonstrate mastery through assessments or publications"
        ]

        if context['user_profile']['intent'] == 'research':
            metrics.append("Contribute to academic publication or conference")
        elif context['user_profile']['intent'] == 'career':
            metrics.append("Develop industry-relevant project portfolio")

        return metrics

class PersonalizedPlanAssembler:


    def assemble_final_plan(self, strategy_context, llm_strategy, topic_info_dict=None):
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Include topic information in metadata if available
        metadata = {
            "generated_date": current_date,
            "query": strategy_context['user_profile']['query'],
            "community": strategy_context['community_profile']['name'],
            "user_intent": strategy_context['user_profile']['intent'],
            "confidence_score": self.calculate_confidence_score(strategy_context)
        }
        
        if topic_info_dict:
            detected_topics = strategy_context['user_profile'].get('detected_topics', [])
            topic_descriptions = []
            for tid in detected_topics:
                if tid in topic_info_dict:
                    topic_info = topic_info_dict[tid]
                    topic_descriptions.append({
                        'topic_id': tid,
                        'name': topic_info['name'],
                        'top_terms': ', '.join(topic_info['top_terms'][:5])
                    })
            metadata['topic_descriptions'] = topic_descriptions
        
        personalized_plan = {
            "metadata": metadata,
            
            "executive_summary": self.generate_executive_summary(strategy_context),
            
            "community_analysis": {
                "key_insights": self.extract_key_insights(strategy_context),
                "strengths_to_leverage": strategy_context['community_profile']['key_strengths'],
                "challenges_to_navigate": strategy_context['community_profile']['weaknesses'],
                "key_opportunities": self.identify_key_opportunities(strategy_context)
            },
            
            "strategic_plan": llm_strategy,
            
            "implementation_guide": {
                "immediate_next_steps": self.generate_immediate_actions(strategy_context),
                "quarterly_goals": self.break_down_timeline(llm_strategy),
                "key_connections": self.prioritize_connections(strategy_context['key_members']),
                "resource_recommendations": self.suggest_resources(strategy_context)
            },
            
            "success_framework": {
                "progress_tracking": self.create_progress_tracking(),
                "adaptation_strategy": self.create_adaptation_plan(),
                "long_term_development": self.suggest_long_term_development()
            }
        }
        
        return personalized_plan

    def calculate_confidence_score(self, context):

        scores = []


        size = context['community_profile']['size']
        if size > 8:
            scores.append(0.8)
        elif size > 3:
            scores.append(0.6)
        else:
            scores.append(0.4)


        quality = context['community_profile']['quality_scores']['research_quality']
        scores.append(quality)


        alignment = context['community_profile']['quality_scores']['topic_alignment']
        scores.append(alignment)

        return round(np.mean(scores), 2)

    def generate_executive_summary(self, context):

        community_name = context['community_profile']['name']
        size = context['community_profile']['size']
        user_goals = context['user_profile']['query']

        return f"""This strategic plan leverages the {community_name} research community ({size} researchers) to help you achieve: "{user_goals}". The plan is tailored to the community's specific strengths and your learning objectives."""

    def extract_key_insights(self, context):

        insights = []

        quality_scores = context['community_profile']['quality_scores']

        if quality_scores['research_quality'] > 0.7:
            insights.append("High-quality research community with strong publication record")
        if quality_scores['collaboration_potential'] > 0.6:
            insights.append("Highly collaborative environment with multiple engagement opportunities")
        if quality_scores['topic_alignment'] > 0.8:
            insights.append("Excellent alignment between community expertise and your interests")

        return insights

    def identify_key_opportunities(self, context):
        opportunities = []
        key_members = context['key_members']

        if key_members.get('leaders'):
            opportunities.append(f"Leadership guidance from {len(key_members['leaders'])} community leaders")
        if key_members.get('bridges'):
            opportunities.append(f"Cross-community connections through {len(key_members['bridges'])} bridge members")
        if key_members.get('experts'):
            opportunities.append(f"Deep expertise from {len(key_members['experts'])} domain specialists")

        return opportunities

    def generate_immediate_actions(self, context):

        return [
            "Review community analysis and key member profiles",
            "Identify 2-3 initial connections based on strategic priorities",
            "Begin foundational learning based on the recommended path",
            "Set up progress tracking for the first month's goals"
        ]

    def break_down_timeline(self, llm_strategy):

        return {
            "Quarter 1": "Foundation building and initial network establishment",
            "Quarter 2": "Skill development and project initiation",
            "Quarter 3": "Advanced topics and collaboration deepening",
            "Quarter 4": "Specialization and contribution to community"
        }

    def prioritize_connections(self, key_members):

        prioritized = []


        for leader in key_members.get('leaders', [])[:2]:
            prioritized.append({
                "researcher": leader['id'],
                "role": "Community Leader",
                "priority": "High",
                "rationale": "Strategic guidance and research direction"
            })


        for bridge in key_members.get('bridges', [])[:2]:
            prioritized.append({
                "researcher": bridge['id'],
                "role": "Bridge Member",
                "priority": "Medium",
                "rationale": "Cross-community connections and opportunities"
            })

        return prioritized

    def suggest_resources(self, context):

        primary_topic = context['user_profile']['detected_topics'][0] if context['user_profile'][
            'detected_topics'] else "AI"
        topic_name = context['user_profile'].get('topic_names', {}).get(primary_topic, "AI")

        resources = {
            "foundational": [
                f"Key textbooks and survey papers in {topic_name}",
                "Online courses covering fundamental concepts",
                "Community-recommended reading lists"
            ],
            "advanced": [
                "Recent research papers from community members",
                "Specialized tutorials and workshops",
                "Open-source projects for practical experience"
            ],
            "community": [
                "Relevant academic conferences and seminars",
                "Online forums and discussion groups",
                "Research group websites and publications"
            ]
        }

        return resources

    def create_progress_tracking(self):

        return {
            "monthly_checkpoints": [
                "Skill acquisition milestones",
                "Network growth metrics",
                "Project completion status"
            ],
            "success_indicators": [
                "Active participation in community discussions",
                "Successful project implementations",
                "Growing professional network"
            ]
        }

    def create_adaptation_plan(self):

        return {
            "evaluation_points": ["Month 3", "Month 6", "Month 9"],
            "adaptation_triggers": [
                "Changing research interests",
                "New community opportunities",
                "Skill development pace adjustments"
            ]
        }

    def suggest_long_term_development(self):

        return {
            "year_1": "Foundation building and community integration",
            "year_2": "Specialization and meaningful contributions",
            "year_3+": "Leadership and mentoring opportunities"
        }