from src.openai.query import completion_with_backoff_mcopenai

def main():
    persona = "You are an assistant who extract information in sentences. Extract the triplets which contains the knowledge of the sentence.\nHere is one example. \n"
    demonstration = "For sentence: Bill Gates and Paul Allen are the founders of Microsoft, your output should be,\n (Bill Gates, found, Microsoft),(Paul Allen, found, Microsoft). \n"
    requirement = "Your response triplets should strictly follow the format: (Subject, Relation, Subject). Note that a sentence may include information of multiple triplets, and you need to divide them with comma and no extra space in your response. Do not include any other words except for the triplets in your response."

    user = "Here is the sentence for you to extract triplets: "
    memory = "Alice and Bob married last year, and they will have a baby next month."

    messages = [
        {"role": "system", "content": persona + demonstration + requirement},
        {"role": "system", "content": user + memory}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    print(response)
if __name__ == "__main__":
    main()