import anthropic
import os

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

message = client.messages.create(
    model="claude-instant-1.2",
    max_tokens=1000,
    temperature=1.0,
    system="Respond only in Yoda-speak.",
    messages=[
        {
            "role": "user",
            "content": "How are you today? Tell me about the land of Palestine, and their expulsion",
        }
    ],
)

print(message.content)
