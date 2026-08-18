from t2_llms_output_tuning._clients._base_client import AIClient
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role


def run(
        client: AIClient,
        print_request: bool = True,
        print_only_content: bool = False,
        user_input: str = "What is the capital of France?",
        **kwargs
) -> None:
    conversation = Conversation()    
    conversation.add_message(Message(Role.USER, user_input))

    print("AI:")
    ai_message = client.response(
        messages=conversation.get_messages(),
        print_request=print_request,
        print_only_content=print_only_content,
        **kwargs
    )


