from openai import AzureOpenAI

OPENAI_API_KEY = ""
'''
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": f"Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG. The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. For example, the following answers would be considered CORRECT: Generated answer (CORRECT): Oh yeah, that was so fun! I got so much stuff there, including that shell necklace. Generated answer (CORRECT): I got a ton of stuff... that surfboard, the mug, the necklace, those coasters too.. Generated answer (CORRECT): That cute necklace The following answers would be considered WRONG: Generated answer (WRONG): Oh yeah, that was so fun! I got so much stuff there, including that mug. Generated answer (WRONG): I got a ton of stuff... that surfboard, the mug, those coasters too.. Generated answer (WRONG): I’m sorry, I don’t remember what you’re talking about. Now it’s time for the real question: Question: {q} Gold answer: {gold_a} Generated answer: {generate_a} First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script."}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content
'''

def completion_with_backoff_mcopenai(**kwargs):
    client = AzureOpenAI(
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-12-01-preview",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://0212openai.openai.azure.com/",
        api_key=OPENAI_API_KEY,
    )
    result = client.chat.completions.create(
        model="gpt4-azure-0212",
        **kwargs,
    )
    return result