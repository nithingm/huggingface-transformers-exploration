from transformers import pipeline, Conversation
import gradio as gr

chatbot = pipeline(task = "conversational", model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def simple_chatbot(message, history):
    past_user_inputs = [entry['content'] for entry in history if entry['role'] == 'user']
    generated_responses = [entry['content'] for entry in history if entry['role'] == 'assistant']

    # Create Conversation object with history
    conversation = Conversation(
        text=message,
        past_user_inputs=past_user_inputs,
        generated_responses=generated_responses
    )

    # Generate a response
    conversation = chatbot(conversation)

    # Return latest response
    return conversation.generated_responses[-1]
    
demo_chatbot = gr.ChatInterface(simple_chatbot, title="Chatbot Uno", description="Enter text and start chatting!")

demo_chatbot.launch(share= True)