import time
import uuid

from django.apps import apps
from django.http import JsonResponse
from rest_framework.decorators import api_view


@api_view(["POST"])
def chat_completions(request):
    messages = request.data.get("messages")
    if not messages:
        return JsonResponse({"error": "Messages are required."}, status=400)

    # Access the globally loaded RejectionSampler instance
    sampler = apps.get_app_config("sotopia").rejection_sampler
    top_response = sampler.inference(messages)

    if top_response is not None:
        # Format the response to mimic OpenAI's chat completion response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "custom-rejection-sampler-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": top_response
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        return JsonResponse(response, status=200)
    else:
        return JsonResponse({"error": "No sample met the threshold."}, status=200)
