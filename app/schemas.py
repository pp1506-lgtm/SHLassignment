"""
schemas.py
Pydantic models for the POST /chat request and response.
Schema is non-negotiable per assignment specification.
"""
from typing import Optional
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 8:
            raise ValueError("messages cannot exceed 8 turns")
        return v


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]  # empty [] when not recommending
    end_of_conversation: bool
