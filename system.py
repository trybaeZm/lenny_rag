from pathlib import Path
import os


DEFAULT_SYSTEM_PROMPT = (
    "You are Lenny, the AI business assistant for Inxource. Use clear, concise, "
    "actionable guidance tailored to SMEs. If the Lenny prompt file is missing, "
    "follow the structure: Insight, Explanation, Recommended Action, and include "
    "a Fact-Check List."
)


def get_system_prompt() -> str:
    """Return the system prompt for the RAG AI.

    Priority:
    1) Environment variable LENNY_PROMPT_PATH if set
    2) Repository default at <repo_root>/lib/lenny_system.md
    3) Fallback to DEFAULT_SYSTEM_PROMPT
    """
    # 1) Env override
    env_path = os.getenv("LENNY_PROMPT_PATH")
    if env_path:
        try:
            return Path(env_path).read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # 2) Default path relative to this file: ../../.. up to repo root "flutter-fastapi-app"
    # system.py is at backend/rag_backend/system.py → repo_root is parents[2]
    try:
        repo_root = Path(__file__).resolve().parents[2]
        default_path = repo_root / "lib" / "lenny_system.md"
        if default_path.exists():
            return default_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass

    # 3) Fallback
    return DEFAULT_SYSTEM_PROMPT

