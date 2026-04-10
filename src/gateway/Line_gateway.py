from src.service.Line_Chat import reply_message
from src.service.LLM_logic import AgentsicAI

def Handle_line_webhook(event: dict) -> tuple[str, Exception]:
    # print(f"Received LINE webhook event: {event}")
    events = event.get("events", [])
    for event in events:
        event_type = event.get("type")
        if event_type == "message":
            reply_token = event.get("replyToken")
            user_message = event["message"].get("text")
            print(f"User message: {user_message}")
            if user_message:
                AgentsicAI_response, err = AgentsicAI(user_message)
                if err is not None:
                    reply_message(reply_token, f"Error: {str(err)}")
                    return None, err
                else:
                    reply_message(reply_token, AgentsicAI_response)
                    return AgentsicAI_response, None