"""QA Bot — Technical interview simulation mode.

The bot acts as an interviewer: asks questions, evaluates answers,
gives feedback, and tracks the session score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .llm_client import ask_llm

# ---------------------------------------------------------------------------
# Interview levels
# ---------------------------------------------------------------------------
LEVELS = {
    "junior": "Junior QA Engineer (0-1 year experience)",
    "middle": "Middle QA Engineer (1-3 years experience)",
    "senior": "Senior QA Engineer (3+ years experience)",
}

LEVEL_KEYBOARD = [
    [("👶 Junior", "junior")],
    [("🧑‍💻 Middle", "middle")],
    [("🧙 Senior", "senior")],
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
INTERVIEWER_SYSTEM = """Ты — технический интервьюер, проводящий собеседование на позицию {level}.

Правила:
1. Задавай ОДИН вопрос за раз, не несколько.
2. После ответа кандидата — дай короткую оценку (1-2 предложения): что хорошо, что можно улучшить.
3. Затем задай следующий вопрос.
4. Вопросы должны быть разнообразными: теория, практика, кейсы.
5. Уровень сложности — {level}.
6. После 5 вопросов подведи итог: оцени кандидата, укажи сильные и слабые стороны.
7. Отвечай на русском языке.
8. Будь строгим но справедливым, как настоящий интервьюер.

Ты ведёшь счёт сессии (0-10 баллов) и в конце объявляешь его."""

FIRST_QUESTION_PROMPT = """Начни интервью. Представься как интервьюер (1 предложение), 
затем задай первый вопрос для позиции {level}. 
Начни с базового вопроса чтобы разогреть кандидата."""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class InterviewSession:
    """Tracks one interview session for a user."""
    level: str
    history: list[dict] = field(default_factory=list)
    question_count: int = 0
    is_finished: bool = False
    MAX_QUESTIONS: int = 5

    @property
    def system_prompt(self) -> str:
        level_desc = LEVELS.get(self.level, self.level)
        return INTERVIEWER_SYSTEM.format(level=level_desc)

    def start(self) -> str:
        """Generate the opening message + first question."""
        level_desc = LEVELS.get(self.level, self.level)
        prompt = FIRST_QUESTION_PROMPT.format(level=level_desc)
        response = ask_llm(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            max_tokens=500,
        )
        self.history.append({"role": "assistant", "content": response})
        self.question_count = 1
        return response

    def answer(self, user_answer: str) -> str:
        """Process user answer, get feedback + next question or final summary."""
        self.history.append({"role": "user", "content": user_answer})

        if self.question_count >= self.MAX_QUESTIONS:
            # Ask for final summary
            self.history.append({
                "role": "user",
                "content": (
                    "Это был последний ответ. Пожалуйста, подведи итог интервью: "
                    "оцени кандидата по шкале 0-10, укажи сильные и слабые стороны, "
                    "и дай рекомендации по развитию."
                ),
            })
            response = ask_llm(
                messages=self.history,
                system=self.system_prompt,
                max_tokens=800,
            )
            self.history.append({"role": "assistant", "content": response})
            self.is_finished = True
            return f"📊 *Итоги интервью:*\n\n{response}"

        # Regular: evaluate answer + next question
        prompt = (
            f"Кандидат ответил: «{user_answer}»\n\n"
            f"Дай краткую оценку этого ответа (1-2 предложения), "
            f"затем задай вопрос №{self.question_count + 1} из {self.MAX_QUESTIONS}."
        )
        self.history.append({"role": "user", "content": prompt})

        response = ask_llm(
            messages=self.history,
            system=self.system_prompt,
            max_tokens=600,
        )
        self.history.append({"role": "assistant", "content": response})
        self.question_count += 1
        return response
