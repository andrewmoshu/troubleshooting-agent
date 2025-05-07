from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Optional, List, Literal
import re

# --- Pydantic Models for Structured LLM Outputs ---
class InitialPlan(BaseModel):
    action: Literal["search", "clarify", "answer"] = Field(description="The optimal first step to take.")
    reasoning: str = Field(description="Brief explanation for the chosen action.")
    queries: Optional[List[str]] = Field(default_factory=list, description="Initial search queries if action is 'search'.")
    clarification_question: Optional[str] = Field(default=None, description="Question to ask the user if action is 'clarify'.")

class QueryComplexityScores(BaseModel):
    complexity: int = Field(description="Rate from 1 (simple) to 5 (very complex).")
    specificity: int = Field(description="Rate from 1 (very general) to 5 (very specific).")
    technical_detail: int = Field(description="Rate from 1 (low detail) to 5 (highly technical).")

    @validator('*', pre=True, allow_reuse=True)
    def ensure_int(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            if isinstance(v, str):
                match = re.search(r'\d+', v)
                if match:
                    return int(match.group(0))
            try:
                return int(float(v))
            except:
                 return 3
        return v

class ContextEvaluation(BaseModel):
    sufficiency_score: float = Field(description="Rate from 0.0 (not sufficient) to 1.0 (fully sufficient).")
    information_gaps: List[str] = Field(default_factory=list, description="Specific pieces of information missing to fully answer the question.")

class RCA(BaseModel):
    root_causes: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    fix_steps: List[str] = Field(default_factory=list)
    commands_to_run: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    additional_queries: List[str] = Field(default_factory=list)
    confidence: float = 0.0

class LinkRelevance(BaseModel):
    url: str = Field(description="The URL that was evaluated.")
    is_relevant: bool = Field(description="True if the link content is likely relevant to the user's question, False otherwise.")
    reasoning: str = Field(description="Brief explanation for the relevance assessment (why it is or isn't relevant).")
    confidence: float = Field(description="Confidence in the relevance assessment (0.0 to 1.0).", ge=0.0, le=1.0) 