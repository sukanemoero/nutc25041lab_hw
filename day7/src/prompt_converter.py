from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

def convert_messages_to_prompt(messages: list[BaseMessage]):
    formatted_list = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        
        formatted_list.append({"role": role, "content": msg.content})

    return tokenizer.apply_chat_template(
        formatted_list, 
        tokenize=False, 
        add_generation_prompt=True
    )
