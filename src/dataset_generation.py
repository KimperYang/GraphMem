import anthropic
import json

client = anthropic.Anthropic(
    api_key="",
)

prompt = """
i want to create a conversation including two people, the conversation should include memory update. the aim of the conversation is to evaluate the current method for agent memory retrieval. first we need to construct a background (or we say, the agent memory) between two people A and B, then based on the background, there exists another people C directly asking some relevant memory about the memory. A or B need to retrieve the relevant memory to continue the conversation. 

please also remember to have explicit memory update part. for example, A's status is X, and then the status updated to Y. the query can ask what is A's status. and your queries should ask the updated memory. But make sure the overall conversation is coherent and logical.

please return a json format like:

{
  "memories": {
    "2024-01-15": [
      {"Alice": "Hey Bob! I just got a new job offer in Seattle. I'm currently working in New York but thinking of moving."},
      {"Bob": "That's exciting! I actually visited Seattle last month. The tech scene is great there. By the way, I just adopted a golden retriever puppy named Max."},
      {"Alice": "Oh wow, congrats on the puppy! I'm still deciding about Seattle, but I have to give my answer by end of January."},
      {"Bob": "Thanks! Max is 3 months old and already so energetic. Let me know if you want to hear more about Seattle."}
    ],
    ......
  },
  "queries": [
    {
      "question asked by Cindy": "Hi Bob! Where does Alice currently live now?",
      "obj": "Bob",
      "answer": "Seattle",
      "reason": "While A initially lived in New York, the memory was updated on 2024-02-01 when A accepted the Seattle job, and further confirmed on 2024-03-15 when A mentioned being settled in Seattle."
    },
    ......
  ]
}
"""

data = []

from tqdm import tqdm
for i in tqdm(range(100)):
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = message.content[0].text
    json_data = json.loads(content)
    data.append(json_data)

    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)