import gradio as gr
from huggingface_hub import InferenceClient
import time
from typing import Generator, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv
import requests
import uuid
from gtts import gTTS
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORY_THEMES = [
    "Adventure",
    "Mystery",
    "Romance",
    "Historical",
    "Slice of Life",
    "Fairy Tale"
]

CHARACTER_TEMPLATES = {
    "Adventurer": "A brave and fearless explorer who loves adventure and challenges.",
    "Detective": "A keen and observant detective skilled in observation and deduction.",
    "Artist": "A creative artist with unique perspectives on beauty.",
    "Scientist": "A curious scientist dedicated to exploring the unknown.",
    "Ordinary Person": "An ordinary person with a rich inner world."
}

STORY_STYLES = [
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Adventure",
    "Romance",
    "Horror"
]

STORY_SYSTEM_PROMPT = """You are a professional story generator. Your task is to generate coherent and engaging stories based on user settings and real-time input.

Key requirements:
1. Maintain continuity with previous plot developments
2. Integrate new user inputs logically
3. Keep character personalities consistent
4. Make stories vivid and engaging

You should not:
1. Start a new story abruptly
2. Contradict previous plot points
"""

MAX_RETRIES = 3
RETRY_DELAY = 2

# Output directories
STORIES_DIR = os.path.join(os.getcwd(), "stories")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(STORIES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def create_client() -> InferenceClient:
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        return InferenceClient(model_name, token=hf_token)
    else:
        return InferenceClient(model_name)


def generate_story(
    scene: str,
    style: str,
    theme: str,
    character_desc: str,
    history: list = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    if history is None:
        history = []

    # Build context
    story_content = [msg["content"] for msg in history if msg.get("role") == "assistant"]
    context_summary = "\n".join(["Previously in the story:", "---"] + story_content + ["---"]) if story_content else ""

    if not history:
        prompt = f"""
Please start a story based on the following settings:

Style: {style}
Theme: {theme}
Character: {character_desc}
Initial Scene: {scene}

Begin from this scene and set up the story's opening.
"""
    else:
        prompt = f"""
{context_summary}

Story settings reminder:
- Style: {style}
- Theme: {theme}
- Main Character: {character_desc}

User's new input: {scene}

Continue the story based on previous plot and user's input.
"""

    messages = [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        client = create_client()
        response = ""

        for message in client.chat_completion(messages, max_tokens=max_tokens, stream=True, temperature=temperature, top_p=top_p):
            # The InferenceClient streaming objects differ depending on version;
            # we attempt to read the text delta safely.
            try:
                if message.choices and hasattr(message.choices[0].delta, 'content'):
                    token = message.choices[0].delta.content
                else:
                    token = None
            except Exception:
                token = None

            if token:
                response += token
                yield response

    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        yield f"Sorry, encountered an error while generating the story: {str(e)}\nPlease try again later."


def save_story(chatbot, style=None, theme=None, character_desc=None):
    if not chatbot:
        return "Story is empty, cannot save"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(STORIES_DIR, f"story_{timestamp}.txt")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== Interactive Story ===\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if style:
                f.write(f"Style: {style}\n")
            if theme:
                f.write(f"Theme: {theme}\n")
            if character_desc:
                f.write(f"Character: {character_desc}\n")
            f.write("\n=== Story Content ===\n\n")
            for i, (user_msg, ai_msg) in enumerate(chatbot, 1):
                f.write(f"--- Turn {i} ---\n")
                if user_msg:
                    f.write(f"User: {user_msg}\n")
                if ai_msg:
                    f.write(f"AI: {ai_msg}\n")
                f.write("\n")
        return gr.Markdown(f"✅ Story saved successfully to: {os.path.basename(filename)}")
    except Exception as e:
        logger.error(f"Error saving story: {str(e)}")
        return gr.Markdown(f"❌ Failed to save story: {str(e)}")


# --- New helper functions for image, audio and pdf generation ---

def get_full_story_text(chatbot) -> str:
    """
    Concatenate the current chatbot conversation into a single story string.
    Prioritize assistant messages but include user prompts for context.
    """
    if not chatbot:
        return ""
    parts = []
    for u, a in chatbot:
        if u:
            parts.append(u.strip())
        if a:
            parts.append(a.strip())
    return "\n\n".join(parts).strip()


def hf_text_to_image(prompt: str, model: str = None, hf_token: str = None) -> tuple:
    """
    Generate an image via Hugging Face Inference HTTP API.
    Returns (image_path_or_None, error_message_or_None).
    - model: model slug (e.g., "stabilityai/stable-diffusion-2"). If None, uses HF_IMAGE_MODEL env var or fallback.
    - hf_token: optional HF token (defaults to HF_TOKEN env var).
    """
    import requests, uuid, time, os, logging
    logger = logging.getLogger(__name__)

    model = model or os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-2")
    hf_token = hf_token or os.getenv("HF_TOKEN")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    logger.info("HF image request URL: %s", url)
    logger.info("HF image request model: %s", model)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            content_type = resp.headers.get("Content-Type", "")
            logger.info("HF image API attempt %s status=%s content-type=%s", attempt, resp.status_code, content_type)

            # If raw image bytes are returned
            if resp.status_code == 200 and content_type and content_type.startswith("image/"):
                img_bytes = resp.content
                img_filename = os.path.join(OUTPUTS_DIR, f"image_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                with open(img_filename, "wb") as f:
                    f.write(img_bytes)
                logger.info("Saved generated image to %s", img_filename)
                return img_filename, None

            # Try parse JSON error response
            try:
                j = resp.json()
            except ValueError:
                j = None

            # Helpful logging and handling
            if resp.status_code == 404:
                # Model not found
                logger.error("HF image API 404: model not found at %s", url)
                # If HF returns JSON error, include it
                if j and "error" in j:
                    return None, f"Model not found: {j.get('error')}"
                return None, "Model not found (404). Check your model slug (HF_IMAGE_MODEL) or whether the model exists."

            if j and isinstance(j, dict) and "error" in j:
                err = j.get("error")
                logger.warning("HF image generation error JSON: %s", err)
                # If model is loading, retry
                if "loading" in err.lower() or "model is loading" in err.lower():
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        return None, "Model is still loading after retries."
                # Gated / permission errors will be visible here
                return None, f"Hugging Face error: {err}"

            # Other non-200 responses
            logger.warning("HF image generation unexpected response: status=%s body=%s", resp.status_code, resp.text[:1000])
            return None, f"Unexpected HF response status={resp.status_code}. See server logs."

        except requests.RequestException as e:
            logger.error("Request exception during HF image generation attempt %s: %s", attempt, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return None, f"Request error: {e}"

    return None, "Exceeded max retries for HF image generation."


def generate_image_action(chatbot, style, theme, character_desc, last_files: Dict[str, Any]):
    """
    Trigger image generation based on the current story or character description.
    Returns (image_path or None, updated last_files dict, status message)
    """
    story_text = get_full_story_text(chatbot)
    if not story_text:
        prompt = f"{style} {theme} illustration of {character_desc}"
    else:
        excerpt = (story_text[:800] + "...") if len(story_text) > 800 else story_text
        prompt = f"Illustration in {style} style, theme: {theme}. Character: {character_desc}. Scene: {excerpt}"

    hf_token = os.getenv("HF_TOKEN")
    image_path, error_msg = hf_text_to_image(prompt, hf_token=hf_token)

    if image_path and os.path.exists(image_path):
        last_files = dict(last_files or {})
        last_files["image"] = image_path
        # Return actual image file path so Gradio can display it
        return image_path, last_files, f"✅ Image generated and displayed successfully!"
    else:
        err_text = error_msg or "❌ Failed to generate image. Check HF_TOKEN, model availability, or network."
        return None, last_files, err_text



def generate_tts_action(chatbot, last_files: Dict[str, Any]):
    """
    Convert the full story to audio (mp3) using gTTS. Returns (audio_path, updated last_files, status).
    """
    story_text = get_full_story_text(chatbot)
    if not story_text:
        return None, last_files, "Story is empty, nothing to convert to audio."

    # gTTS has limits in practice for a single call; we trim to a reasonable size or split.
    # We'll use the first ~4000 characters for a single mp3 file to avoid extremely long TTS processing.
    tts_text = story_text.strip()
    max_chars = 4000
    if len(tts_text) > max_chars:
        tts_text = tts_text[:max_chars] + "..."

    try:
        tts = gTTS(text=tts_text, lang="en")
        audio_filename = os.path.join(OUTPUTS_DIR, f"story_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp3")
        tts.save(audio_filename)
        last_files = dict(last_files or {})
        last_files["audio"] = audio_filename
        return audio_filename, last_files, f"✅ Audio generated and saved to {os.path.basename(audio_filename)}"
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None, last_files, f"❌ Failed to generate audio: {e}"


def export_pdf_action(chatbot, last_files: Dict[str, Any], style=None, theme=None, character_desc=None):
    """
    Export the current story to a PDF. Optionally embed the last generated image if present.
    Returns (pdf_path, updated last_files, status).
    """
    story_text = get_full_story_text(chatbot)
    if not story_text:
        return None, last_files, "Story is empty, cannot export to PDF."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(OUTPUTS_DIR, f"story_{timestamp}.pdf")
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Header
        pdf.cell(0, 8, "Interactive Story", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        if style:
            pdf.cell(0, 6, f"Style: {style}", ln=True)
        if theme:
            pdf.cell(0, 6, f"Theme: {theme}", ln=True)
        if character_desc:
            pdf.cell(0, 6, f"Character: {character_desc}", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", size=11)
        # Insert image if exists
        image_path = (last_files or {}).get("image")
        if image_path and os.path.exists(image_path):
            try:
                # Fit image width with some margin
                pdf.image(image_path, x=15, w=pdf.w - 30)
                pdf.ln(6)
            except Exception as e:
                logger.warning(f"Failed to insert image into PDF: {e}")

        # Add story text as multi-cell
        # FPDF multi_cell will wrap lines automatically
        pdf.multi_cell(0, 6, txt=story_text)
        pdf.output(pdf_filename)
        last_files = dict(last_files or {})
        last_files["pdf"] = pdf_filename
        return pdf_filename, last_files, f"✅ PDF exported to {os.path.basename(pdf_filename)}"
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}")
        return None, last_files, f"❌ Failed to export PDF: {e}"


# --- Gradio UI with new buttons and outputs ---

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet",secondary_hue="blue",neutral_hue="gray")) as demo:
        gr.Markdown("# 🎭 Interactive Story Generator\nLet AI create unique stories!")
        with gr.Tabs():
            with gr.Tab("✍️ Story Creation"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        style_select = gr.Dropdown(choices=STORY_STYLES, value="Fantasy", label="Story Style")
                        theme_select = gr.Dropdown(choices=STORY_THEMES, value="Adventure", label="Story Theme")
                        character_select = gr.Dropdown(choices=list(CHARACTER_TEMPLATES.keys()), value="Adventurer", label="Character Template")
                        character_desc = gr.Textbox(lines=3, value=CHARACTER_TEMPLATES["Adventurer"], label="Character Description")
                        scene_input = gr.Textbox(lines=3, placeholder="Describe the scene...", label="Scene Description")
                        submit_btn = gr.Button("✨ Start Story", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear Chat")
                        save_btn = gr.Button("💾 Save Story")

                        gr.Markdown("### Export / Media")
                        generate_image_btn = gr.Button("🖼️ Generate Image")
                        generate_audio_btn = gr.Button("🔊 Text → Audio")
                        export_pdf_btn = gr.Button("📄 Export PDF")
                        # Hidden state to keep track of last files (image/audio/pdf)
                        last_files_state = gr.State({"image": None, "audio": None, "pdf": None})

                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Story Dialogue", height=600)
                        status_msg = gr.Markdown("")

                        # New outputs
                        image_out = gr.Image(label="Generated Image", interactive=False)
                        audio_out = gr.Audio(label="Generated Audio", interactive=False)
                        pdf_out = gr.File(label="Download PDF")

            with gr.Tab("⚙️ Advanced Settings"):
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Creativity (Temperature)")
                max_tokens = gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Max Generation Length")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Sampling Range (Top-p)")

        def update_character_desc(template):
            return CHARACTER_TEMPLATES.get(template, "")

        character_select.change(update_character_desc, character_select, character_desc)

        def user_input(user_message, history):
            if history is None:
                history = []
            if not user_message or not user_message.strip():
                return "", history
            history.append([user_message, None])
            return "", history

        def bot_response(history, style, theme, character_desc, temperature, max_tokens, top_p):
            if not history or not history[-1][0]:
                yield history
                return
            try:
                user_message = history[-1][0]
                message_history = []
                for u, b in history[:-1]:
                    if u:
                        message_history.append({"role": "user", "content": u})
                    if b:
                        message_history.append({"role": "assistant", "content": b})

                current_response = ""
                for text in generate_story(user_message, style, theme, character_desc, message_history, temperature, max_tokens, top_p):
                    current_response = text
                    history[-1][1] = current_response
                    yield history
            except Exception as e:
                logger.error(f"Error in bot_response: {str(e)}")
                history[-1][1] = "Sorry, encountered an error while generating the story."
                yield history

        def clear_chat():
            return [], ""

        # Wire up core story generation interactions
        submit_btn.click(user_input, [scene_input, chatbot], [scene_input, chatbot]).then(
            bot_response, [chatbot, style_select, theme_select, character_desc, temperature, max_tokens, top_p], chatbot
        )
        scene_input.submit(user_input, [scene_input, chatbot], [scene_input, chatbot]).then(
            bot_response, [chatbot, style_select, theme_select, character_desc, temperature, max_tokens, top_p], chatbot
        )
        clear_btn.click(clear_chat, None, [chatbot, status_msg])
        save_btn.click(save_story, [chatbot, style_select, theme_select, character_desc], status_msg)

        # Image generation button
        generate_image_btn.click(
            fn=generate_image_action,
            inputs=[chatbot, style_select, theme_select, character_desc, last_files_state],
            outputs=[image_out, last_files_state, status_msg],
        )

        # Text to audio button
        generate_audio_btn.click(
            fn=generate_tts_action,
            inputs=[chatbot, last_files_state],
            outputs=[audio_out, last_files_state, status_msg],
        )

        # Export PDF button (uses last image if any)
        export_pdf_btn.click(
            fn=export_pdf_action,
            inputs=[chatbot, last_files_state, style_select, theme_select, character_desc],
            outputs=[pdf_out, last_files_state, status_msg],
        )

        return demo

if __name__ == "__main__":
    import os

    # Create the Gradio demo
    demo = create_demo()

    # Render provides a dynamic port; use 7860 as fallback for local runs
    port = int(os.environ.get("PORT", 7860))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )