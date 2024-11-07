from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import time
import uuid

@csrf_exempt
def chat_completions(request):
    if request.method == "POST":
        # Parse JSON request body
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        
        # Extract model and messages from the request body
        model = data.get("model", "gpt-4o-mini")
        messages = data.get("messages", [])
        
        # Simulate a response (replace this part with actual API call or LLM processing as needed)
        response_content = "This is a test!"  # Stub response text
        created_time = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:6]}"

        # Mock response structure
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model,
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }

        return JsonResponse(response, status=200)
    else:
        return JsonResponse({"error": "Invalid HTTP method"}, status=405)
