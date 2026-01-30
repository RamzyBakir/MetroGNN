
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