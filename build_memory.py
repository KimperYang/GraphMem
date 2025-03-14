import json
import re
from src.openai.query import completion_with_backoff_mcopenai
from src.graph.knowledge_graph import SemanticKnowledgeGraph
import pdb
from tqdm import tqdm
import time
def get_triplet(memory):
    # kg = SemanticKnowledgeGraph()
    sys_trip = """
    You are an assistant who extract information in daily dialog sentences. Extract the triplets (Subject, Relation, Subject) which contains the knowledge of the sentence.\nHere are two examples.

    For sentence: "Jack: Hi! I just passed my final exam.", the extracted triplets should be ("Jack", "passed", "final exam")
    For sentence: "Alice: Hi Bob, do you want to come to my 24 year old birthday party tomorrow?", the extracted triplets should be ("Bob", "invited", "Alice's birthday party"),("Alice", "24", "age")

    Your response triplets should strictly follow the format: (Subject, Relation, Subject). Note that a sentence may include information of multiple triplets, and you need to divide them with comma and no extra space in your response. Do not include any other words except for the triplets in your response.
    """

    user = f"Here is the sentence for you to extract triplets: {memory}"

    messages = [
        {"role": "system", "content": sys_trip},
        {"role": "user", "content": user}
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

            """

    messages = [
        {"role": "system", "content": sys_ref},
        {"role": "user", "content": prompt_template}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    return response

def process_question(question):
    sys_q = """
    You are an assistant who extract information in daily dialog question. Extract the triplets which contains the knowledge of the sentence.
    However, since this is a question, the triplet must be not complete in the forms like (Subject, Relation, Unknown), (Subject, Unknown, Subject), or (Unknown, Relation, Subject)
    Here are two examples.

    For sentence: "question asked by Carol: Do you know how old is Alice?", the extracted triplets should be ("Alice", "Unknown", "age")
    For sentence: "question asked by Carol: John, where are you working right now?", the extracted triplets should be ("John", "work", "Unknown")

    Use the Unknown in the triplets to refer to the entity which is questioned in the question. Do not include any other words except for the triplets in your response.
    """

    user = f"Here is the sentence for you to extract triplets: {question}"
    messages = [
        {"role": "system", "content": sys_q},
        {"role": "user", "content": user}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content

    return response

def get_agent_response(retrieved, question):

    sys_r = """
    **Role:** You are an assistant answering daily dialog questions based on provided memory triplets `(Subject, Relation, Object)`. These triplets represent your knowledge in a structured format. For example, the triplet `("Alice", "Study", "Biology")` means Alice is studying Biology, and `("Bob", "24", "age")` means Bob is 24 years old.

    **Instructions:**  
    - Answer questions based on these triplets, which are sorted by relevance (most important first).  
    - If the answer is not explicitly stated, infer the answer by making logical deductions based on the triplets.  
    - If you cannot determine an answer, provide a reasonable random answer based on the triplets.  
    - Directly provide the final answer without explaining your reasoning unless explicitly asked.
    """

    for i in range(len(retrieved)):
        retrieved[i] = str(retrieved[i])

    retrieved = "\n".join(retrieved)

    user = f"""
    Knowledge triplets:{retrieved}.

    Question: {question}
    """

    messages = [
        {"role": "system", "content": sys_r},
        {"role": "user", "content": user}
    ]

    response = completion_with_backoff_mcopenai(messages = messages, temperature = 0, max_tokens=50).choices[0].message.content
    print(response)
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
    return triplets

def parse_llm_judge_response(response: str) -> bool:

    match = re.search(r'Cover:\s*(True|False)', response)
    if match:
        return match.group(1) == "True"
    else:
        print(response)
        raise ValueError("LLM response does not contain a valid 'Cover:' line.")

def main():
    data_path = 'data/data.json'
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_num = 0
    correct_num = 0
    total_time = 0
    # Build Graph
    for idx in range(20):
        try:
            entry = data[idx]
            kg = SemanticKnowledgeGraph()
            memories = entry.get("memories", {})
            queries = entry.get("queries", [])

            for date, conversation_list in memories.items():
                for message in conversation_list:
                    for person, text in message.items():
                        extracted_mems = extract_triplets(get_triplet(f"{person}: {text}"))
                        for mem in extracted_mems:
                            res = kg.add_edge(mem[0], mem[1], mem[2], False)
                            try:
                                if res['conflict'] and parse_llm_judge_response(reflection(mem, res['message'])):
                                    kg.add_edge(mem[0], mem[1], mem[2], True)
                            except ValueError as e:
                                print(f"Error: {e}")
                                continue
            
            kg.draw()
            
            for query in queries:

                question = query.get("question", "")
                g_answer = query.get("answer", "")
                print(question, g_answer)
                total_num += 1
                extracted_q = extract_triplets(process_question(f"{question}"))
                for q in extracted_q:
                    retrieved = []
                    if "Unknown" in q[0]:
                        retrieved += kg.query(node1=None, relation=q[1], node2=q[2],top_k=3,use_gnn=True,max_hops=2)

                    elif "Unknown" in q[1]:
                        retrieved += kg.query(node1=q[0], relation=None, node2=q[2], top_k=3,use_gnn=True,max_hops=5)
                        
                    elif "Unknown" in q[2]:
                        retrieved += kg.query(node1=q[0], relation=q[1] ,node2=None, top_k=3,use_gnn=True,max_hops=2)
                        
                    else:
                        print(f"Warning: At least one unknown entity needed. {q}")
                    
                print(question, extracted_q)
                print('--'*20)
                print(retrieved)
                start_time = time.time()
                response = get_agent_response(retrieved, question)
                end_time = time.time()
                total_time += end_time - start_time
                if parse_llm_judge_response(llm_judge(question, g_answer, response)):
                    correct_num += 1
                    
                print(f"Total number of questions: {total_num}")
                print(f"Number of correct answers: {correct_num}")
                print(f"Accuracy: {correct_num/total_num}")
                print(f"Average response time: {total_time}")
                        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
if __name__ == "__main__":
    main()