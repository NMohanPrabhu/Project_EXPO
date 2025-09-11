import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Load Model ---
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # Good CPU model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

translator = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)  # CPU

# --- Language Codes for NLLB ---
languages = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Arabic": "arb_Arab",
    "Chinese (Simplified)": "zho_Hans",
    "Hindi": "hin_Deva",
}

# --- Translation Function ---
def translate_text(text, source_lang, target_lang):
    if not text.strip():
        return "⚠️ Please enter some text."
    src = languages[source_lang]
    tgt = languages[target_lang]
    result = translator(text, src_lang=src, tgt_lang=tgt)
    return result[0]['translation_text']

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌍 Language Translator App  
        Type your text, choose source and target languages, and click **Translate**.
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Enter Text", lines=5, placeholder="Type here...")
            source_lang = gr.Dropdown(list(languages.keys()), value="English", label="Source Language")
            target_lang = gr.Dropdown(list(languages.keys()), value="French", label="Target Language")
            translate_button = gr.Button("🔄 Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated Text", lines=5)

    translate_button.click(translate_text, inputs=[input_text, source_lang, target_lang], outputs=output_text)

demo.launch()
