import requests
import json

# Define the request details
url = "http://localhost:8005/v1/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "qwen-2.5-instruct",
    "prompt": (
        "Imagine you are Ethan Johnson, your task is to act/speak as Ethan Johnson would, keeping in mind Ethan Johnson's social goal.\n"
        "You can find Ethan Johnson's background and goal in the 'Here is the context of the interaction' field.\n"
        "Note that Ethan Johnson's secret and goal is only visible to you.\n"
        "You should try your best to achieve Ethan Johnson's goal in a way that align with their character traits.\n"
        "Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n\n"
        "Here is the context of this interaction:\n"
        "Scenario: Two friends are on a long road trip. They have different music tastes. Agent1 prefers pop music and enjoys singing along loudly, "
        "while Agent2 prefers classical music and enjoys the quiet contemplation it provides. They need to decide on a playlist that will keep both entertained during the drive.\n"
        "Participants: Ethan Johnson and Benjamin Jackson\n"
        "Ethan Johnson's background: Ethan Johnson is a 34-year-old male chef. He/him pronouns. Ethan Johnson is famous for cooking Italian food. "
        "Personality and values description: Ethan Johnson, a creative yet somewhat reserved individual, values power and fairness. "
        "He likes to analyse situations before deciding. Ethan's secrets: He is a closeted romance novel author.\n"
        "Benjamin Jackson's background: Benjamin Jackson is a 24-year-old male environmental activist. He/him pronouns. Benjamin Jackson is well-known for his impassioned speeches. "
        "Personality and values description: Benjamin Jackson, expressive and imaginative, leans towards self-direction and liberty. His decisions aim for societal betterment. \n"
        "Ethan Johnson's goal: Find a compromise on the road trip playlist (Extra information: Agent1 gets joy from singing along to pop songs and may feel bored or restless without it.)\n"
        "Benjamin Jackson's goal: Unknown\n"
        "Conversation Starts:\n\n"
        "Turn #0: Ethan Johnson said: \"Hey, Benjamin. I know we have different tastes in music. I was thinking, maybe we could switch off between my pop music list and your classical list? "
        "This way, we both could enjoy what we like during the trip.\"\n\n"
        "Turn #1: Benjamin Jackson said: \"Hey Ethan, that sounds like a fair plan! But I'd prefer if we could keep the volume down when we're playing the pop music, "
        "so I can also endure some contemplative moments even when your songs are on. Would that be okay for you?\"\n\n"
        "Turn #2: Ethan Johnson said: \"Sure Benjamin, I completely understand. I'll keep the volume on the lower side when my pop songs are on. "
        "But remember, it's a long trip and I might need to energize myself at times by singing along. How about we keep the volume moderate, so I can still sing but it won't intrude upon your contemplation?\"\n\n"
        "Turn #3: Benjamin Jackson said: \"That's a fair compromise, Ethan. We can maintain a moderate volume during your pop music and I trust that even though you'll sing along, you'll have my interests in mind. "
        "Should we also consider adding some instrumental versions of your pop songs? That way you get your tunes but in a form that suits my classical taste?\"\n\n"
        "Turn #4: Ethan Johnson said: \"You know what, Benjamin? That's a great idea! I really appreciate your suggestion. At times when I'm not singing along to the pop songs, "
        "I could also enjoy the instrumental versions. It would still help me keep the energy up and at the same time, it might give you that classical vibe. "
        "I think we've got ourselves a pretty balanced road trip playlist!\"\n\n"
        "Turn #5: Benjamin Jackson said: \"I'm glad we could agree on this, Ethan! Knowing we'll have a balanced playlist makes the road trip even more exciting. "
        "Music is such a powerful tool, don't you think? It can bring people together, even with different tastes like us. Now, looking forward to our journey!\"\n\n"
        "Turn #6: Ethan Johnson said: \"Absolutely agreed, Benjamin. Music indeed has an incredible power to unify. I think it's part of the reason why I love cooking so much as well - "
        "the right combination of ingredients is no different than a harmonious symphony. I can't wait for our musical adventure to begin!\"\n\n"
        "Turn #7: Benjamin Jackson said: \"I really love that analogy, Ethan, cooking and music are indeed similar in creating harmony. "
        "It will be a beautiful journey with this mix of music. Alright then, let's set off on our journey with our newly minted playlist!\"\n\n"
        "You are at Turn #8. Your available action types are\n"
        "\"none action speak non-verbal communication leave\".\n"
        "Note: You can \"leave\" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, "
        "3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.\n\n"
        "Please only generate a JSON string including the action type and the argument.\n"
        "Your action should follow the given format:\n\n"
        "As an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n"
        "the object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\n"
        "Here is the output schema:\n"
        "{\"description\": \"An interface for messages.\\nThere is only one required method: to_natural_language\", "
        "\"properties\": {\"action_type\": {\"title\": \"Action Type\", \"description\": \"whether to speak at this turn or choose to not do anything\", "
        "\"enum\": [\"none\", \"speak\", \"non-verbal communication\", \"action\", \"leave\"], \"type\": \"string\"}, "
        "\"argument\": {\"title\": \"Argument\", \"description\": \"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\", "
        "\"type\": \"string\"}}, \"required\": [\"action_type\", \"argument\"]}"
    ),
    "max_tokens": 100,
    "temperature": 0
}

# Send the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
import rich
# Print the response
print("Status Code:", response.status_code)
json_data = response.json()
print("Response JSON:", rich.print(json_data["choices"][0]["text"]) if response.status_code == 200 else response.text)