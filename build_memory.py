import json
import re
from src.openai.query import completion_with_backoff_mcopenai
from src.graph.knowledge_graph import SemanticKnowledgeGraph

def get_triplet(memory):
    # kg = SemanticKnowledgeGraph()
    sys_trip = """
    You are an assistant who extract information in daily dialog sentences. Extract the triplets which contains the knowledge of the sentence.\nHere are two examples.

    For sentence: "Jack: Hi! I just passed my final exam.", the extracted triplets should be ("Jack", "passed", "final exam")
    For sentence: "Alice: Hi Bob, do you want to come to my 24 year old birthday party tomorrow?", the extracted triplets should be ("Bob", "invited", "Alice's birthday party"),("Alice", "24", "age")

    Your response triplets should strictly follow the format: (Subject, Relation, Subject). Note that a sentence may include information of multiple triplets, and you need to divide them with comma and no extra space in your response. Do not include any other words except for the triplets in your response.
    """

    user = "Here is the sentence for you to extract triplets: "

    messages = [
        {"role": "system", "content": sys_trip},
        {"role": "system", "content": user + memory}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0.5, max_tokens=50).choices[0].message.content

    return response

def reflection(old, new):
    sys_ref = "You are an assistant that evaluates two triplets of information."
    prompt_template = f"""

            1. The old triplet: {old}
            2. The new triplet: {new}

            Determine if the new triplet should cover the old triplet or if there is no conflict between them.

            - If the new triplet updates, changes, or supersedes the old triplet, print exactly: "Cover: True"
            - If the new triplet does not affect the old one or there is no direct contradiction, print exactly: "Cover: False"

            Explain your reasoning in a short sentence before your final answer.
            """

    messages = [
        {"role": "system", "content": sys_ref},
        {"role": "system", "content": prompt_template}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    return response

def extract_triplets(input_str):
    # Find all occurrences of triplets enclosed in parentheses
    # This will capture strings like: I, focus on, contemporary dance teaching
    matches = re.findall(r'\(([^)]+)\)', input_str)
    
    if not matches:
        
        print(f"Warning: No triplets found in the input string. {input_str}")
        return []

    triplets = []
    for match in matches:
        # Split the captured string by comma
        parts = [part.strip() for part in match.split(',')]
        if len(parts) == 3:
            triplets.append(tuple(parts))
        else:
            # If any captured group doesn't split into exactly three parts,
            # it doesn't match the expected triplet format.
            print(f"Warning: Invalid triplet format encountered. {parts}")
            return []
    print(triplets)
    return triplets

def parse_llm_judge_response(response: str) -> bool:

    match = re.search(r'Cover:\s*(True|False)', response)
    if match:
        return match.group(1) == "True"
    else:
        raise ValueError("LLM response does not contain a valid 'Cover:' line.")

def main():
    kg = SemanticKnowledgeGraph()

    data_path = 'data/data.json'
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build Graph
    for entry in data:
        kg = SemanticKnowledgeGraph()
        memories = entry.get("memories", {})
        queries = entry.get("queries", [])

        for date, conversation_list in memories.items():
            for message in conversation_list:
                for person, text in message.items():
                    extracted_mems = extract_triplets(get_triplet(f"{person}: {text}"))
                    for mem in extracted_mems:
                        res = kg.add_edge(mem[0], mem[2], mem[1], False)
                        if res['conflict'] and parse_llm_judge_response(reflection(mem, res['message'])):
                            kg.add_edge(mem[0], mem[2], mem[1], True)
        
        kg.draw()
        # kg.dump()
        # TODO: Retrieve
        print("Queries:")
        for q in queries:
            question = q.get("question asked by Carol", "")
            print(f"Question: {question}")

if __name__ == "__main__":
    main()