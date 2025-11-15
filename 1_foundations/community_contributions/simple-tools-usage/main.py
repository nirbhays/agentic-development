from dotenv import load_dotenv
from anthropic import Anthropic
import re, json

load_dotenv(override=True)
anthropic_client = Anthropic()

call_to_action = "Type something to manipulate, or 'exit' to quit."

def smart_capitalize(word):
    for i, c in enumerate(word):
        if c.isalpha():
            return word[:i] + c.upper() + word[i+1:].lower()
    return word  # no letters to capitalize

def manipulate_string(input_string):
    input_string = input_string[::-1]
    words = re.split(r'\s+', input_string.strip())
    capitalized_words = [smart_capitalize(word) for word in words]
    return ' '.join(capitalized_words)

manipulate_string_json = {
    "name": "manipulate_string",
    "description": "Use this tool to reverse the characters in the text the user enters, then to capitalize the first letter of each reversed word)",
    "input_schema": {
        "type": "object",
        "properties": {
            "input_string": {
                "type": "string",
                "description": "The text the user enters"
            }
        },
        "required": ["input_string"]
    }
}

tools = [manipulate_string_json]

TOOL_FUNCTIONS = {
    "manipulate_string": manipulate_string
}

def handle_tool_calls(tool_uses):
    results = []
    for tool_use in tool_uses:
        tool_name = tool_use.name
        arguments = tool_use.input
        tool = TOOL_FUNCTIONS.get(tool_name)
        result = tool(**arguments) if tool else {}
        content = result if isinstance(result, str) else json.dumps(result)
        results.append({
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content
        })
    return results

system_prompt = f"""You are a helpful assistant who takes text from the user and manipulates it in various ways.
Currently you do the following:
- reverse the string the user entered
- convert to all lowercase letters so any words whose first letters were capitalized are now lowercase
- convert the first letter of each word in the reversed string to uppercase
Be professional, friendly and engaging, as if talking to a customer who came across your service.
Do not output any additional text, just the result of the string manipulation.
After outputting the text, prompt the user for the next input text with {call_to_action}
With this context, please chat with the user, always staying in character.
"""

def chat(message, history):
    cleaned_history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = cleaned_history + [{"role": "user", "content": message}]
    done = False
    while not done:
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system=system_prompt,
            messages=messages,
            tools=tools
        )
        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            results = handle_tool_calls(tool_uses)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
        else:
            done = True
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return text_blocks[0] if text_blocks else ""

def main():
    print("\nWelcome to the string manipulation chat!")
    print(f"{call_to_action}\n")
    history = []

    while True:
        user_input = input("")
        if user_input.lower() in {"exit", "quit"}:
            print("\nThanks for using our service!")
            break

        response = chat(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        print(response)

if __name__ == "__main__":
    main()
