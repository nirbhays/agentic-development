import os
from anthropic import Anthropic
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY environment variable. Aborting smoke test.")
        return
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=32,
        messages=[{"role": "user", "content": "Respond with the word 'ping' only."}]
    )
    text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
    print('SMOKE_TEST_RESPONSE:', text_blocks[0] if text_blocks else 'NO_TEXT')

if __name__ == "__main__":
    main()
