"""Multi-agent specialists for Session 14.

Each agent is a pure function that receives the state and returns a partial update.
Agents have minimum privilege: each can only access its declared tools.

Agent tool privileges:
- requirements_extractor: NO business tools (LLM only)
- budget_searcher: search_budgets
- estimate_generator: calculate_estimate
- coherence_validator: validate_estimate
"""

from src.domain.multi_agent.agents.budget_searcher import budget_searcher
from src.domain.multi_agent.agents.coherence_validator import coherence_validator
from src.domain.multi_agent.agents.estimate_generator import estimate_generator
from src.domain.multi_agent.agents.requirements_extractor import requirements_extractor

__all__ = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
]
