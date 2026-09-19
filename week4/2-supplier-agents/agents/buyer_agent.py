"""Buyer Agent.

Goal: receive the factory request, identify the decision criteria,
coordinate the workflow, and present the final answer to the user.
It is the only agent that talks to the user.
"""
import re
from agents.supplier_analysis_agent import analyze
from common import ask, banner
from tools.supplier_tool import find_supplier

# ---------------------------------------------------------------------------
# TODO 1  —  Complete this prompt.
#
# As written it asks for a loose summary, so the Supplier Analysis Agent
# receives vague criteria and has to guess the quantity and the deadline.
#
# Rewrite it so the Buyer Agent states the required quantity, the deadline
# in days, and which attributes matter for the decision. A short, fixed
# output shape is easier for the next agent to use than free prose.
# ---------------------------------------------------------------------------
CRITERIA_PROMPT = """You are the Buyer Agent for a factory.

Read the factory request below and summarise what the factory wants.

Factory request:
{request}

Answer in two or three short lines.
"""


def extract_criteria(llm, request: str) -> str:
    banner("Buyer Agent: identifying the decision criteria")
    criteria = ask(llm, CRITERIA_PROMPT.format(request=request))
    print(criteria)
    return criteria


def _extract_quantity_and_deadline(request: str) -> tuple[int | None, int | None]:
    """Pulls the requested quantity (units) and deadline (days) out of the
    factory request text using simple pattern matching.
 
    This is intentionally independent of TODO 1's prompt, so the risk check
    still works even before the Buyer Agent's criteria prompt is improved.
    """
    quantity = None
    deadline = None
 
    qty_match = re.search(r"(\d+)\s*units", request, re.IGNORECASE)
    if qty_match:
        quantity = int(qty_match.group(1))
 
    deadline_match = re.search(r"(\d+)\s*days?", request, re.IGNORECASE)
    if deadline_match:
        deadline = int(deadline_match.group(1))
 
    return quantity, deadline
 
 
def check_risks(recommendation: str, request: str, criteria: str) -> list[str]:
    """Flags a recommendation that cannot actually be met.
 
    Checks the named supplier's structured record against the factory's
    quantity and deadline, and against a minimum reliability threshold.
    """
    warnings: list[str] = []
 
    supplier = find_supplier(recommendation)
    if supplier is None:
        warnings.append(
            "The recommended supplier could not be matched to any supplier "
            "in the catalogue. The recommendation may be invented."
        )
        return warnings
 
    quantity, deadline = _extract_quantity_and_deadline(request)
 
    if quantity is not None and supplier["capacity_units"] < quantity:
        warnings.append(
            f"{supplier['name']}'s capacity ({supplier['capacity_units']} units) "
            f"is below the requested quantity ({quantity} units)."
        )
 
    if deadline is not None and supplier["delivery_days"] > deadline:
        warnings.append(
            f"{supplier['name']}'s delivery time ({supplier['delivery_days']} days) "
            f"exceeds the requested deadline ({deadline} days)."
        )
 
    RELIABILITY_THRESHOLD = 0.75
    if supplier["reliability_score"] < RELIABILITY_THRESHOLD:
        warnings.append(
            f"{supplier['name']}'s reliability score "
            f"({supplier['reliability_score']:.2f}) is below the "
            f"{RELIABILITY_THRESHOLD} threshold considered acceptable."
        )
 
    return warnings


def format_response(recommendation: str, warnings: list[str]) -> str:
    """Builds the answer the user sees.

    -----------------------------------------------------------------------
    TODO 4  —  Improve this format.

    It currently passes the model's raw text straight through. Give the
    user a predictable structure instead: the selected supplier, the
    justification, the trade-offs against the rejected options, and any
    warnings from check_risks.
    -----------------------------------------------------------------------
    """
    parts = [recommendation]
    if warnings:
        parts.append("\nWarnings:")
        parts.extend(f"  - {w}" for w in warnings)
    return "\n".join(parts)


def handle_request(llm, request: str) -> str:
    """The coordination strategy: a sequential handoff, orchestrated here."""
    banner("Factory user  ->  Buyer Agent")
    print(request)

    criteria = extract_criteria(llm, request)
    recommendation = analyze(llm, request, criteria)
    warnings = check_risks(recommendation, request, criteria)

    banner("Buyer Agent  ->  Factory user")
    answer = format_response(recommendation, warnings)
    print(answer)
    return answer
