import json
import re
from src.openai.query import completion_with_backoff_mcopenai
from src.retriever.bm25 import BM25Retriever


def get_agent_response(retrieved, question):
    sys_r = """
    **Role:** You are an assistant answering daily dialog questions based on provided memory.

    **Instructions:**  
    - Answer questions based on these memory, which are sorted by relevance (most important first).  
    - If the answer is not explicitly stated, infer the answer by making logical deductions based on the memory.  
    - If you cannot determine an answer, provide a reasonable random answer based on the memory.
    - Directly provide the final answer without explaining your reasoning unless explicitly asked.
    """
    for i in range(len(retrieved)):
        retrieved[i] = str(retrieved[i])

    retrieved = "\n".join(retrieved)

    user = f"""
    Your retrieved memories:{retrieved}.

    {question}
    """

    messages = [
        {"role": "system", "content": sys_r},
        {"role": "user", "content": user}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    return response

def llm_judge(question, g_answer, response):
    sys_judge = """
    You are an assistant who judge the answer of a question from another assistant. 
    You will be given the question, the ground truth answer, and the assistant's response.
    """

    user = f"""
    Question: {question}
    Ground truth answer: {g_answer}
    Assistant's response: {response}

    - If the you think his response is correct, print exactly: "Cover: True"
    - If the you think his response is not correct, print exactly: "Cover: False"

    Explain your reasoning in a short sentence before your final answer.
    """
    messages = [
        {"role": "system", "content": sys_judge},
        {"role": "user", "content": user}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    return response

def parse_llm_judge_response(response: str) -> bool:

    match = re.search(r'Cover:\s*(True|False)', response)
    if match:
        return match.group(1) == "True"
    else:
        raise ValueError("LLM response does not contain a valid 'Cover:' line.")

def main():
    data_path = 'data/data.json'
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_num = 0
    correct_num = 0
    # Build Graph
    for idx in range(20):
        try:
            entry = data[idx]

            memories = entry.get("memories", {})
            queries = entry.get("queries", [])

            extracted_mems = []
            for date, conversation_list in memories.items():
                for message in conversation_list:
                    for person, text in message.items():
                        extracted_mems.append(f"{person}: {text}")

            retriever = BM25Retriever()
            retriever.fit(extracted_mems)
            
            for q in queries:
                question = q.get("question", "")
                g_answer = q.get("answer", "")
                print(question, g_answer)
                total_num += 1

                retrieved = retriever.retrieve(question, top_k=1)
                print(question, retrieved)

                response = get_agent_response(retrieved, question)

                if parse_llm_judge_response(llm_judge(question, g_answer, response)):
                    correct_num += 1

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"Total number of questions: {total_num}")
    print(f"Number of correct answers: {correct_num}")
    print(f"Accuracy: {correct_num/total_num}")

if __name__ == "__main__":
    main()