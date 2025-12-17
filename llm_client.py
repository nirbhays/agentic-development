from anthropic import Anthropic, AsyncAnthropic
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_MODEL = "claude-3-5-haiku-20241022"

def to_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure messages are in Anthropic format (role/content)."""
    cleaned = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        # If content already list of dicts, leave; else wrap as text block
        if isinstance(content, list):
            cleaned.append({"role": role, "content": content})
        else:
            cleaned.append({"role": role, "content": str(content)})
    return cleaned

def call_anthropic(messages: List[Dict[str, Any]],
                   model: str = DEFAULT_MODEL,
                   max_tokens: int = 512,
                   system: Optional[str] = None,
                   tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Any]:
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=to_anthropic_messages(messages),
        tools=tools,
    )
    # Collect first text block
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return (text_blocks[0] if text_blocks else "", response)

async def call_anthropic_async(messages: List[Dict[str, Any]],
                               model: str = DEFAULT_MODEL,
                               max_tokens: int = 512,
                               system: Optional[str] = None,
                               tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Any]:
    client = AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=to_anthropic_messages(messages),
        tools=tools,
    )
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return (text_blocks[0] if text_blocks else "", response)

def extract_tool_uses(response: Any) -> List[Any]:
    return [b for b in response.content if getattr(b, "type", None) == "tool_use"]
