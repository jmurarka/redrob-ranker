import math
import re
from typing import List, Dict

SENIORITY_MAP = {
    "intern": 0, "trainee": 0, "junior": 1, "associate": 1,
    "mid": 2, "senior": 3, "lead": 4, "principal": 5, "staff": 5,
    "head": 6, "director": 7, "vp": 8, "chief": 9,
}

def _seniority_level(title: str) -> int:
    """Helper function to map a title string to its seniority integer level."""
    t = title.lower()
    best = 2  # default: mid
    for kw, lvl in SENIORITY_MAP.items():
        if kw in t:
            best = max(best, lvl)
    return best

class CareerTrajectoryAnalyzer:
    """
    Analyzes career progression, promotion history, and leadership scope
    from candidate work history records.
    """

    def detect_promotions(self, work_history: List[Dict]) -> int:
        """
        Counts promotion events by detecting title-level increases within the same company.
        
        Args:
            work_history: List of role dictionaries sorted newest to oldest.
            
        Returns:
            int: Number of internal promotion events.
        """
        if not work_history or len(work_history) < 2:
            return 0
        # Reverse to oldest-to-newest chronological order
        chron = list(reversed(work_history))
        promotions = 0
        for i in range(len(chron) - 1):
            c1 = chron[i].get("company", "").strip().lower()
            c2 = chron[i+1].get("company", "").strip().lower()
            t1 = chron[i].get("title", "")
            t2 = chron[i+1].get("title", "")
            
            # Count only if it is the same company and title seniority increased
            if c1 == c2 and c1 != "":
                l1 = _seniority_level(t1)
                l2 = _seniority_level(t2)
                if l2 > l1:
                    promotions += 1
        return promotions

    def compute_leadership_scope(self, work_history: List[Dict]) -> float:
        """
        Scores leadership scope on a 0.0 to 1.0 scale based on management titles,
        team size mentions, and cross-functional collaboration indicators.
        
        Args:
            work_history: List of role dictionaries.
            
        Returns:
            float: Leadership scope score in range [0.0, 1.0].
        """
        if not work_history:
            return 0.0
        
        has_management_title = False
        has_team_mention = False
        has_cross_functional = False
        
        mgmt_words = {"lead", "head", "manager", "director", "vp", "chief", "principal", "staff", "president"}
        cross_words = {"cross-functional", "collaborated", "partnered", "stakeholders", "product manager", "designer", "business unit"}
        
        for role in work_history:
            title = role.get("title", "").lower()
            desc = role.get("description", "").lower()
            
            if any(w in title for w in mgmt_words):
                has_management_title = True
                
            if any(w in desc for w in cross_words):
                has_cross_functional = True
                
            # Regex for team sizes, e.g. "team of 5", "managed 8"
            if re.search(r"\b(?:team\s+of|managed|led|hiring|team\s+size|size\s+of)\s*(?:\s+of)?\s*(\d+|five|six|seven|eight|nine|ten)\b", desc):
                has_team_mention = True
                
        score = 0.0
        if has_management_title:
            score += 0.4
        if has_team_mention:
            score += 0.3
        if has_cross_functional:
            score += 0.3
        return float(score)

    def model_trajectory(self, work_history: List[Dict]) -> float:
        """
        Returns a trajectory score based on recency-weighted role seniority progression.
        
        Args:
            work_history: List of role dictionaries sorted newest to oldest.
            
        Returns:
            float: Trajectory score in range [0.0, 1.0] (0.5 is stable growth).
        """
        if not work_history:
            return 0.0
        
        n = len(work_history)
        if n == 1:
            return float(_seniority_level(work_history[0].get("title", "")) / 9.0)
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i in range(n - 1):
            l_new = _seniority_level(work_history[i].get("title", ""))
            l_old = _seniority_level(work_history[i+1].get("title", ""))
            diff = l_new - l_old
            
            # Recency weight decays for older transitions
            weight = 1.0 / (i + 1)
            weighted_sum += diff * weight
            total_weight += weight
            
        avg_diff = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Map avg_diff to 0.0 - 1.0 via sigmoid
        score = 1.0 / (1.0 + math.exp(-avg_diff))
        return float(score)

    def compute_promotion_velocity(self, work_history: List[Dict]) -> float:
        """
        Calculates the promotion rate (velocity) per year of career history.
        """
        promotions = self.detect_promotions(work_history)
        total_months = sum(role.get("duration_months", 0) for role in work_history)
        total_years = max(0.5, total_months / 12.0)
        return float(promotions / total_years)

    def compute_specialization_consistency(self, work_history: List[Dict]) -> float:
        """
        Calculates consistency of roles aligning with core technical titles.
        """
        if not work_history:
            return 0.0
        tech_words = {"ml", "ai", "machine learning", "deep learning", "nlp", "data scien", 
                      "recommendation", "search", "ranking", "retrieval", "llm", "engineer", 
                      "developer", "architect", "programmer"}
        specialized = 0
        for role in work_history:
            title = role.get("title", "").lower()
            if any(w in title for w in tech_words):
                specialized += 1
        return float(specialized / len(work_history))

    def compute_leadership_growth(self, work_history: List[Dict]) -> float:
        """
        Measures the growth of leadership and seniority responsibility over time.
        """
        if not work_history:
            return 0.0
        levels = [_seniority_level(r.get("title", "")) for r in work_history]
        if not levels:
            return 0.0
        # Seniority progression delta (latest vs minimum)
        sen_delta = (levels[0] - min(levels)) / 9.0
        leadership_growth = min(1.0, max(0.0, sen_delta))
        
        # Add bonus if latest role holds leadership titles
        recent_title = work_history[0].get("title", "").lower()
        if any(w in recent_title for w in ["lead", "head", "manager", "director", "vp", "chief", "principal"]):
            leadership_growth = min(1.0, leadership_growth + 0.3)
        return float(leadership_growth)

    def compute_tenure_stability(self, work_history: List[Dict]) -> float:
        """
        Evaluates career stability based on average role durations.
        """
        if not work_history:
            return 0.0
        avg_months = sum(role.get("duration_months", 0) for role in work_history) / len(work_history)
        if avg_months >= 24:
            return 1.0
        elif avg_months <= 12:
            return 0.3
        # Linear interpolation between 12 and 24 months
        return float(0.3 + 0.7 * (avg_months - 12) / 12)

    def compute_domain_focus_score(self, work_history: List[Dict]) -> float:
        """
        Scores domain breadth and alignment with specialized industry keywords.
        """
        if not work_history:
            return 0.0
        domain_kws = {"saas", "fintech", "healthcare", "hr-tech", "adtech", "ai", "e-commerce", 
                      "enterprise", "cloud", "security", "gdpr", "hipaa", "soc2"}
        hits = 0
        all_text = " ".join(r.get("description", "").lower() for r in work_history) + " " + " ".join(r.get("title", "").lower() for r in work_history)
        for kw in domain_kws:
            if kw in all_text:
                hits += 1
        return float(min(1.0, hits / 4.0))


def calculate_trajectory_slope(career_history: List[Dict]) -> float:
    """Calculates linear slope of seniority levels over career timeline in years."""
    career_chron = list(reversed(career_history))
    if len(career_chron) < 2:
        return 0.0
    
    t_points = []
    l_points = []
    
    cumulative_months = 0.0
    for role in career_chron:
        duration = role.get("duration_months", 0)
        role_midpoint_months = cumulative_months + (duration / 2.0)
        t_points.append(role_midpoint_months / 12.0)
        l_points.append(_seniority_level(role.get("title", "")))
        cumulative_months += duration
        
    n = len(t_points)
    sum_x = sum(t_points)
    sum_y = sum(l_points)
    sum_xy = sum(x * y for x, y in zip(t_points, l_points))
    sum_xx = sum(x * x for x in t_points)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if abs(denominator) < 1e-5:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

def calculate_growth_score(career_history: List[Dict], has_production_evidence: bool, skills: List[Dict] = None) -> float:
    """Calculates overall growth potential score based on slope, promotions, and skill velocity."""
    if not career_history:
        return 0.2
        
    slope = calculate_trajectory_slope(career_history)
    try:
        norm_slope = 1.0 / (1.0 + math.exp(-3.0 * (slope - 0.3)))
    except Exception:
        norm_slope = 0.5
        
    total_months = sum(role.get("duration_months", 0) for role in career_history)
    total_years = max(0.5, total_months / 12.0)
    
    # Instantiate the analyzer to count promotions
    analyzer = CareerTrajectoryAnalyzer()
    promotions = analyzer.detect_promotions(career_history)
    
    promo_velocity = promotions / total_years
    promo_index = min(1.0, promo_velocity / 0.5)
    
    num_skills = len(skills) if skills else 0
    skill_velocity = num_skills / total_years
    skill_index = min(1.0, skill_velocity / 3.0)
    
    growth_score = 0.40 * norm_slope + 0.30 * promo_index + 0.30 * skill_index
    return min(1.0, max(0.0, growth_score))
