import aiohttp
import asyncio

async def send_prompt(session, prompt):
    
    data = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192
    }
    url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}

    try:
        # Send the POST request asynchronously
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                return await response.json()  # Parse the JSON response
            else:
                print(f"Error for prompt '{prompt}': Received status code {response.status}")
                print(f"Response: {await response.text()}")
                return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def send_batch_prompts(prompts):

    async with aiohttp.ClientSession() as session:
        # Create a list of tasks for sending all prompts asynchronously
        tasks = [
            send_prompt(session, prompt)
            for prompt in prompts
        ]

        # Wait for all tasks to complete (barrier synchronization)
        results = await asyncio.gather(*tasks)

    return results

def make_batch_query(prompts):
    results = asyncio.run(send_batch_prompts(prompts))
    output_texts = []
    for result in results:
        if result and "text" in result:  # Check if the response is valid
            output_texts.append(result["text"][0])
        else:
            output_texts.append("")  # Handle errors gracefully
    return output_texts
