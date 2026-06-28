"""
LangGraph ReAct agent for Ford vehicle intelligence.
"""

import logging
import os
import re
from types import SimpleNamespace
from typing import Optional

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - depends on environment
    ChatGroq = None

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover - depends on environment
    create_react_agent = None
    MemorySaver = None

from agent.tools import (
    search_vehicle_complaints,
    search_vehicle_complaints_raw,
    get_recall_summary,
    get_recall_summary_raw,
    get_complaint_stats,
    get_complaint_stats_raw,
)

SYSTEM_PROMPT = """You are a Ford vehicle safety intelligence assistant powered by NHTSA public data.

You help users understand:
- Common complaint patterns for specific Ford models and years
- Active recall campaigns and their remedies
- Which vehicle components have the most reported issues

Always cite the source (NHTSA complaint/recall data) in your responses.

When a user asks which models or years are supported, answer directly without calling tools:
- Supported models: Bronco, Explorer, Mustang, Escape, Edge
- Supported years: 2020 – 2024
- Note that live NHTSA data is always available even when a model/year isn't yet indexed locally.

When a user asks about a specific model or year, ALWAYS follow this strategy:
1. Call get_complaint_stats to get the live complaint breakdown by component.
2. Call search_vehicle_complaints to find relevant complaint summaries from the local database.
3. Call get_recall_summary if the user asks about recalls or safety campaigns.
4. If search_vehicle_complaints returns no local results, rely on get_complaint_stats for the answer — do NOT tell the user there are no complaints.

FORMATTING RULES — follow exactly:
- When presenting stats across multiple vehicle models, use one "### MODEL NAME (YEAR)" markdown heading per model, then a markdown bullet list beneath it.
- Each bullet must be its own line: "- **COMPONENT NAME**: N complaints"
- Preserve NHTSA category names in their original uppercase (e.g., POWER TRAIN, ENGINE AND ENGINE COOLING, ELECTRICAL SYSTEM).
- Never collapse a multi-model breakdown into a single run-on paragraph.
- For a single model with multiple components, still use a bullet list — one component per line.
- If the knowledge base has no relevant data, explicitly note that the information is not available in the local knowledge base at the moment, then continue with relevant NHTSA live data.
- Flag safety-critical issues (crashes, fires, rollaways) with ⚠ on the same bullet line.
- Always end with a brief plain-language summary paragraph after the structured data.

Be concise but thorough. Available Ford models: Bronco, Explorer, Mustang, Escape, Edge (years 2020-2024).
"""

_MAX_TOKENS = 5_500  # stay well under Groq's 12k request limit

_agent = None
_checkpointer = None


class _FallbackAgent:
    """
    Rule-based responder used when the LangGraph/Groq stack is rate-limited.
    Covers every meaningful query class so no user ever receives a dead-end reply.
    """

    # ── message extraction ──────────────────────────────────────────────────

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        messages = state.get("messages", [])
        user_text = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
            else:
                content = getattr(last, "content", "")
            if isinstance(content, list):
                user_text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                user_text = str(content)
        response = self._build_response(user_text)
        return {"messages": [SimpleNamespace(type="ai", content=response)]}

    # ── extraction helpers ───────────────────────────────────────────────────

    def _extract_model_year(self, text: str) -> tuple[Optional[str], Optional[int]]:
        lower = (text or "").lower()
        model_match = re.search(r"\b(bronco|explorer|mustang|escape|edge)\b", lower)
        year_match = re.search(r"\b(20(?:20|21|22|23|24))\b", lower)
        model = model_match.group(1) if model_match else None
        year = int(year_match.group(1)) if year_match else None
        return model, year

    def _extract_all_models(self, text: str) -> list:
        return list(dict.fromkeys(
            re.findall(r"\b(bronco|explorer|mustang|escape|edge)\b", text.lower())
        ))

    def _extract_any_year(self, text: str) -> Optional[int]:
        """Captures any plausible year (1900–2099), for out-of-range detection."""
        m = re.search(r"\b((?:19|20)\d{2})\b", text)
        return int(m.group(1)) if m else None

    # ── intent detectors ─────────────────────────────────────────────────────

    def _is_greeting(self, text: str) -> bool:
        return bool(re.match(
            r"^(hi|hello|hey|howdy|greetings|good\s+(?:morning|afternoon|evening))[!.,\s]*$",
            text.lower()
        ))

    def _is_thanks(self, text: str) -> bool:
        return bool(re.match(
            r"^(thanks?|thank\s+you|cheers|ok|okay|got\s+it|perfect|great|sounds\s+good|cool)[!.,\s]*$",
            text.lower()
        ))

    def _is_recall_query(self, text: str) -> bool:
        return bool(re.search(
            r"\b(recall|recalls|campaign|safety\s+campaign|open\s+recall|recall\s+campaign|nhtsa\s+campaign)\b",
            text.lower()
        ))

    def _is_complaint_query(self, text: str) -> bool:
        return bool(re.search(
            r"\b(complaint|complaints|issue|issues|problem|problems|failure|failures|fault|faults|defect|defects)\b",
            text.lower()
        ))

    def _is_stats_query(self, text: str) -> bool:
        return bool(re.search(
            r"\b(stat|statistics|how\s+many|breakdown|most\s+common|top|frequent|number\s+of|count|percentage)\b",
            text.lower()
        ))

    def _is_comparison_query(self, text: str) -> bool:
        return bool(re.search(
            r"\b(vs\.?|versus|compare|comparison|better|worse|difference|between)\b",
            text.lower()
        ))

    def _is_meta_query(self, text: str) -> bool:
        patterns = [
            r"\b(which|what)\b.{0,30}\b(model|year|vehicle|car|supported|available|covered)\b",
            r"\b(model|year|vehicle)s?\b.{0,30}\b(support|available|covered|have|data)\b",
            r"\b(supported|coverage)\b",
            r"\b(list|show|tell\s+me).{0,20}\b(model|year|vehicle)\b",
            r"\bwhat\s+data\b",
            r"\bwhat.{0,15}(know|have|database|knowledge\s+base)\b",
        ]
        lower = text.lower()
        return any(re.search(p, lower) for p in patterns)

    def _is_help_query(self, text: str) -> bool:
        return bool(re.search(
            r"\b(help|what\s+can\s+you|what\s+do\s+you|tell\s+me\s+about\s+yourself|what\s+are\s+you|capabilities|features)\b",
            text.lower()
        ))

    # ── input validation ─────────────────────────────────────────────────────

    def _validate_query(self, text: str, model: Optional[str], year: Optional[int]) -> Optional[str]:
        # Non-Ford brand mentioned
        non_ford = re.search(
            r"\b(toyota|honda|chevy|chevrolet|bmw|mercedes|volkswagen|vw|nissan|hyundai|"
            r"kia|jeep|ram|dodge|chrysler|gmc|buick|cadillac|tesla|rivian|lucid|audi|volvo|subaru)\b",
            text.lower()
        )
        if non_ford:
            brand = non_ford.group(1).replace("chevy", "chevrolet").title()
            return (
                f"This assistant only covers Ford vehicles. "
                f"I don't have data for {brand}.\n"
                "Supported Ford models: **Bronco, Explorer, Mustang, Escape, Edge** (years 2020–2024)."
            )

        # Year out of supported range
        any_year = self._extract_any_year(text)
        if any_year is not None and (any_year < 2020 or any_year > 2024):
            return (
                f"**{any_year}** is outside the supported range (2020–2024). "
                "This assistant covers NHTSA safety data for Ford model years 2020 through 2024. "
                "Please try a year within that range."
            )

        # Unsupported Ford model explicitly named
        if model and model.lower() not in {"bronco", "explorer", "mustang", "escape", "edge"}:
            return (
                f"'{model.title()}' is not in the supported model list. "
                "Supported Ford models: **Bronco, Explorer, Mustang, Escape, Edge** (years 2020–2024)."
            )

        return None

    # ── response handlers ────────────────────────────────────────────────────

    def _handle_greeting(self) -> str:
        return (
            "Hello! I'm the **Ford Vehicle Safety Intelligence** assistant, powered by NHTSA public data.\n\n"
            "I can help you with:\n"
            "- **Complaint patterns** — common owner-reported issues by component\n"
            "- **Open recalls** — active NHTSA safety campaigns and remedies\n"
            "- **Component breakdowns** — which systems have the highest failure rates\n"
            "- **Model comparisons** — side-by-side complaint stats\n\n"
            "**Try asking:**\n"
            "- *'What are the top issues on a 2022 Bronco?'*\n"
            "- *'Any open recalls on the 2023 Explorer?'*\n"
            "- *'Compare Bronco and Mustang complaints for 2022'*\n\n"
            "**Supported:** Bronco, Explorer, Mustang, Escape, Edge — model years **2020–2024**."
        )

    def _handle_general(self) -> str:
        return (
            "I'm the **Ford Vehicle Safety Intelligence** assistant. Here's what I can help with:\n\n"
            "**Complaint Analysis**\n"
            "Ask about specific complaint patterns, common failure modes, and owner-reported issues.\n"
            "_Example: 'What are the most common problems with the 2022 Ford Explorer?'_\n\n"
            "**Open Recalls**\n"
            "Look up active NHTSA safety recall campaigns and their remedies.\n"
            "_Example: 'Are there any open recalls on the 2023 Bronco?'_\n\n"
            "**Component Breakdowns**\n"
            "See which vehicle systems (power train, electrical, brakes, etc.) rank highest in complaints.\n"
            "_Example: 'What components fail most on the 2021 Mustang?'_\n\n"
            "**Model Comparisons**\n"
            "Compare two models side by side on complaint counts.\n"
            "_Example: 'Compare Bronco vs Explorer in 2022'_\n\n"
            "**Coverage:** Bronco, Explorer, Mustang, Escape, Edge — model years **2020–2024**."
        )

    def _handle_recall(self, model: Optional[str], year: Optional[int]) -> str:
        if model and year:
            return get_recall_summary_raw(model=model, year=year)

        if model and not year:
            # Try most-populated years first; stop on first meaningful result
            for try_year in [2023, 2022, 2024, 2021, 2020]:
                result = get_recall_summary_raw(model=model, year=try_year)
                if not any(skip in result for skip in [
                    "not recognise", "Error fetching", "outside the supported", "timed out"
                ]):
                    return (
                        f"_No year was specified — showing recalls for **{model.title()} {try_year}**. "
                        f"Ask for a specific year to narrow results._\n\n{result}"
                    )
            return (
                f"No open recalls found for Ford {model.title()} across supported model years (2020–2024). "
                "This may mean there are currently no active campaigns, or the model/year is not in NHTSA's system."
            )

        if year and not model:
            return (
                f"To look up recalls for **{year}**, please also specify the Ford model.\n"
                "Supported models: **Bronco, Explorer, Mustang, Escape, Edge**.\n"
                f"_Example: 'Any open recalls on the {year} Explorer?'_"
            )

        return (
            "To look up recalls I need at least a **model name**.\n"
            "Supported models: Bronco, Explorer, Mustang, Escape, Edge (years 2020–2024).\n"
            "_Example: 'Any open recalls on the 2023 Bronco?'_"
        )

    def _handle_complaint(self, model: Optional[str], year: Optional[int], query: str) -> str:
        if model and year:
            try:
                return search_vehicle_complaints_raw(query=query, model=model, year=year)
            except Exception:
                logging.exception("KB search failed; falling back to live stats")
                return get_complaint_stats_raw(model=model, year=year)

        if model and not year:
            # Search KB across all years for this model
            try:
                kb = search_vehicle_complaints_raw(query=query, model=model, year=None)
            except Exception:
                kb = None

            # Complement with live stats for the most populated year
            stats_note = ""
            for try_year in [2023, 2022, 2024, 2021, 2020]:
                stats = get_complaint_stats_raw(model=model, year=try_year)
                if not any(skip in stats for skip in [
                    "No complaint", "not recognise", "Error fetching", "timed out"
                ]):
                    stats_note = (
                        f"\n\n**Live NHTSA complaint stats for {model.title()} {try_year}:**\n{stats}"
                    )
                    break

            if kb:
                return kb + stats_note
            if stats_note:
                return stats_note.strip()
            return f"No complaint data found for Ford {model.title()} in the supported year range (2020–2024)."

        if year and not model:
            # Use Explorer as a representative model for year-only queries
            result = get_complaint_stats_raw(model="explorer", year=year)
            if not any(skip in result for skip in [
                "No complaint", "not recognise", "Error fetching", "timed out"
            ]):
                return (
                    f"_No model was specified — showing complaint breakdown for **Explorer {year}** as a reference. "
                    f"Specify a model for a different vehicle._\n\n{result}"
                )
            return (
                f"To look up complaints for **{year}**, please also specify the Ford model.\n"
                "Supported models: **Bronco, Explorer, Mustang, Escape, Edge**.\n"
                f"_Example: 'What are the issues with the {year} Explorer?'_"
            )

        # No model or year — broad KB search
        try:
            return search_vehicle_complaints_raw(query=query, model=None, year=None)
        except Exception:
            return self._handle_general()

    def _handle_comparison(self, text: str) -> str:
        models = self._extract_all_models(text)
        _, year = self._extract_model_year(text)

        if len(models) < 2:
            return (
                "To compare vehicles, please name **two Ford models** in your question.\n"
                "Supported models: Bronco, Explorer, Mustang, Escape, Edge.\n"
                "_Example: 'Compare the 2022 Bronco and Explorer'_"
            )

        compare_year = year or 2023
        year_note = f" for **{compare_year}**" + ("" if year else " _(defaulting to 2023 — specify a year to change this)_")
        sections = []
        for mdl in models[:2]:
            stats = get_complaint_stats_raw(model=mdl, year=compare_year)
            sections.append(f"### {mdl.title()} {compare_year}\n{stats}")

        return f"**Side-by-side complaint comparison{year_note}:**\n\n" + "\n\n".join(sections)

    def _handle_vehicle(self, model: Optional[str], year: Optional[int], query: str) -> str:
        """General vehicle query — show KB complaints + live stats."""
        if model and year:
            try:
                kb = search_vehicle_complaints_raw(query=query, model=model, year=year)
            except Exception:
                kb = None
            stats = get_complaint_stats_raw(model=model, year=year)
            parts = [p for p in [kb, stats] if p and "unavailable" not in p.lower()]
            return "\n\n".join(parts) if parts else f"No data found for Ford {model.title()} {year}."

        if model:
            return self._handle_complaint(model, None, query)
        if year:
            return self._handle_complaint(None, year, query)
        return self._handle_general()

    def _get_coverage_response(self) -> str:
        from agent.tools import _get_collection
        from collections import defaultdict

        lines = [
            "This assistant covers NHTSA vehicle safety data for Ford vehicles.\n",
            "**Supported models:** Bronco, Explorer, Mustang, Escape, Edge",
            "**Year range:** 2020 – 2024\n",
        ]
        collection = _get_collection()
        if collection and collection.count() > 0:
            try:
                all_meta = collection.get(include=["metadatas"])
                by_model: dict = defaultdict(set)
                for m in all_meta["metadatas"]:
                    mdl = m.get("model") or "unknown"
                    yr = m.get("year")
                    if yr:
                        by_model[mdl].add(int(yr))
                lines.append("**Currently indexed in local knowledge base:**")
                for mdl in sorted(by_model):
                    years_sorted = sorted(by_model[mdl])
                    lines.append(f"  - {mdl.title()}: {', '.join(str(y) for y in years_sorted)}")
                lines.append("")
            except Exception:
                logging.exception("Failed to read KB coverage")

        lines.append(
            "Live NHTSA complaint and recall data is also available for **any** Ford model/year "
            "even if it isn't yet indexed locally.\n"
        )
        lines.append(
            "Try: *'What are the top issues on a 2022 Bronco?'* or *'Any open recalls on the 2023 Explorer?'*"
        )
        return "\n".join(lines)

    # ── main dispatch ─────────────────────────────────────────────────────────

    def _build_response(self, user_text: str) -> str:
        text = (user_text or "").strip()

        if not text:
            return self._handle_general()

        if self._is_greeting(text):
            return self._handle_greeting()

        if self._is_thanks(text):
            return "You're welcome! Let me know if you have any other Ford vehicle safety questions."

        model, year = self._extract_model_year(text)

        # Input validation — non-Ford brands, out-of-range years
        validation_error = self._validate_query(text, model, year)
        if validation_error:
            return validation_error

        # Coverage / meta queries
        if self._is_meta_query(text) and not model and not year:
            return self._get_coverage_response()

        # Comparison queries
        if self._is_comparison_query(text) and len(self._extract_all_models(text)) >= 2:
            return self._handle_comparison(text)

        # Recall queries
        if self._is_recall_query(text):
            return self._handle_recall(model, year)

        # Complaint / stats queries
        if self._is_complaint_query(text) or self._is_stats_query(text):
            return self._handle_complaint(model, year, text)

        # Help / capability queries
        if self._is_help_query(text):
            return self._handle_general()

        # General vehicle query with a model or year mentioned
        if model or year:
            return self._handle_vehicle(model, year, text)

        return self._handle_general()


def _est_tokens(msgs) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    total = 0
    for m in msgs:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(content) // 4
    return total


def _trim_hook(state: dict) -> dict:
    """Pre-model hook: trim oldest messages to stay under token budget."""
    messages = state.get("messages", [])
    before = _est_tokens(messages)
    logging.debug(f"[DIAG] pre-model: {len(messages)} messages, ~{before} tokens")

    if before <= _MAX_TOKENS:
        return {"messages": messages}

    trimmed = list(messages)
    while len(trimmed) > 1 and _est_tokens(trimmed) > _MAX_TOKENS:
        trimmed.pop(0)

    after = _est_tokens(trimmed)
    logging.debug(f"[DIAG] trimmed to: {len(trimmed)} messages, ~{after} tokens")
    return {"messages": trimmed}


def _get_groq_api_keys() -> list[str]:
    keys: list[str] = []

    env_list = os.getenv("GROQ_API_KEYS", "")
    if env_list:
        keys.extend(k.strip() for k in env_list.split(",") if k.strip())

    for idx in range(1, 6):
        name = "GROQ_API_KEY" if idx == 1 else f"GROQ_API_KEY_{idx}"
        raw_key = os.getenv(name)
        if raw_key and raw_key.strip() and raw_key.strip() not in keys:
            keys.append(raw_key.strip())

    return keys


def _extract_error_text(exc: Exception) -> str:
    text = str(exc) or ""
    if hasattr(exc, "__cause__") and exc.__cause__:
        text += " " + str(exc.__cause__)
    if hasattr(exc, "__context__") and exc.__context__:
        text += " " + str(exc.__context__)
    return text.lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    text = _extract_error_text(exc)
    return any(token in text for token in [
        "429",
        "rate_limit_exceeded",
        "rate limit",
        "quota exceeded",
        "quota",
    ])


def _is_org_level_quota_error(exc: Exception) -> bool:
    text = _extract_error_text(exc)
    return any(token in text for token in [
        "tokens per day",
        "service tier",
        "organization `org_",
        "org_",
    ])


def _build_groq_llm(api_key: str):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
        max_retries=2,
    )


class _GroqAgentWithKeyRotation:
    def __init__(
        self,
        keys: list[str],
        tools,
        prompt: str,
        pre_model_hook,
        checkpointer,
    ):
        self.keys = keys
        self.tools = tools
        self.prompt = prompt
        self.pre_model_hook = pre_model_hook
        self.checkpointer = checkpointer
        self.agent = self._create_agent_for_key(keys[0]) if keys else None

    def _create_agent_for_key(self, api_key: str):
        llm = _build_groq_llm(api_key)
        return create_react_agent(
            llm,
            tools=self.tools,
            prompt=self.prompt,
            pre_model_hook=self.pre_model_hook,
            checkpointer=self.checkpointer,
        )

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        last_error = None
        for key_index, api_key in enumerate(self.keys):
            # Always rebuild so we start fresh with key 1 on every invocation,
            # not whatever key was left over from the last rotation cycle.
            agent = self._create_agent_for_key(api_key)
            if key_index > 0:
                logging.warning(
                    "Groq rate limit triggered; retrying with backup key %d/%d (env key index %d).",
                    key_index + 1,
                    len(self.keys),
                    key_index + 1,
                )
            else:
                logging.info(
                    "Using Groq key %d/%d for initial request.",
                    key_index + 1,
                    len(self.keys),
                )

            try:
                return agent.invoke(state, config=config)
            except Exception as e:
                last_error = e
                if not _is_rate_limit_error(e) or key_index == len(self.keys) - 1:
                    raise

        if last_error:
            raise last_error


def get_fallback_agent():
    return _FallbackAgent()


def get_agent():
    global _agent, _checkpointer
    if _agent is None:
        if ChatGroq is None or create_react_agent is None or MemorySaver is None:
            _checkpointer = None
            _agent = _FallbackAgent()
            return _agent

        try:
            _checkpointer = MemorySaver()
            tools = [search_vehicle_complaints, get_recall_summary, get_complaint_stats]
            groq_keys = _get_groq_api_keys()
            if groq_keys:
                _agent = _GroqAgentWithKeyRotation(
                    keys=groq_keys,
                    tools=tools,
                    prompt=SYSTEM_PROMPT,
                    pre_model_hook=_trim_hook,
                    checkpointer=_checkpointer,
                )
            else:
                _agent = _FallbackAgent()
        except Exception:
            _checkpointer = None
            _agent = _FallbackAgent()
    return _agent
