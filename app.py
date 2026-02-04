import gradio as gr
from ollama import Client

MODEL = "llama3.2:3b"  # Small/fast; upgrade to llama3.1:8b for detailed history
client = Client()     # Assumes Ollama running locally

# Load system prompt
with open("system_prompt.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

def history_chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    response = ""
    stream = client.chat(
        model=MODEL,
        messages=messages,
        stream=True,
    )
    
    for chunk in stream:
        content = chunk['message']['content']
        response += content
        yield response

demo = gr.ChatInterface(
    fn=history_chat,
    title="American History AI",
    description=(
        "An educational tool for discussing American history from pre-colonial to present day. "
        "International relations limited to impacts on America. Not for non-American topics."
    ),
    examples=[
        "What was the significance of the Boston Tea Party?",
        "Explain the Civil War's causes.",
        "How did the Great Depression affect America?",
        "Current US political system overview.",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
