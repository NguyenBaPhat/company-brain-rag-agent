"""WebSocket wire protocol (client -> server). Server -> client events are built in `events.py`.

Client messages
---------------
{"type": "user_message", "text": "...", "language": "en" | "vi", "message_id": "..."}
{"type": "cancel"}
{"type": "ping"}

Server events (all JSON, always carry "type")
---------------------------------------------
session      {session_id, history[], model, llm_mode, languages[]}   sent once after connect
turn_start   {message_id}
token        {delta}                      streamed text chunk of the answer being generated
tool_call    {id, name, args}             agent decided to call a tool
tool_result  {id, name, status, summary}  tool finished (status: ok | error | rejected | ...)
sources      {chunks[]}                   passages retrieved by a search (for citation chips)
final        {text, guardrails}           authoritative answer text (replaces streamed tokens)
error        {code, message}
turn_end     {message_id, usage}
pong         {}
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class UserMessage(BaseModel):
    type: Literal["user_message"]
    text: str = Field(min_length=1)
    language: Literal["en", "vi"] = "en"
    message_id: str | None = Field(default=None, max_length=64)


class CancelMessage(BaseModel):
    type: Literal["cancel"]


class PingMessage(BaseModel):
    type: Literal["ping"]


ClientMessage = Annotated[UserMessage | CancelMessage | PingMessage, Field(discriminator="type")]
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
