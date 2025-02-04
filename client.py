from openai import OpenAI
client = OpenAI(
    api_key="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"

)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a virtual assistant named Marcus skilled in general tasks like Alexa and google cloud."},
        {
            "role": "user", "content": "what is coding."
        }
    ]
)

print(completion.choices[0].message.content)