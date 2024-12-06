from openai import AzureOpenAI

def completion_with_backoff_mcopenai(**kwargs):
    client = AzureOpenAI(
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-12-01-preview",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://0212openai.openai.azure.com/",
        api_key="352d7f1511084d6d8a37f7214c5eb528",
    )
    result = client.chat.completions.create(
        model="gpt4-azure-0212",
        **kwargs,
    )
    return result