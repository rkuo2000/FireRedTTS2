import re
import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal, List, Tuple
from fireredtts2.fireredtts2 import FireRedTTS2


# ================================================
#                   FireRedTTS2 Model
# ================================================
# Global model instance
model: FireRedTTS2 = None


def initiate_model(pretrained_dir: str, device="cuda:1"):
    global model
    if model is None:
        model = FireRedTTS2(
            pretrained_dir=pretrained_dir,
            gen_type="monologue",
            device=device,
        )


# ================================================
#                   Gradio
# ================================================

# i18n
_i18n_key2lang_dict = dict(
    # Title markdown
    title_md_desc=dict(
        en="FireRedTTS-2 🔥 Monologue Generation",
        zh="FireRedTTS-2 🔥 單人語音生成",
    ),
    # Voice mode radio
    voice_mode_label=dict(
        en="Voice Mode",
        zh="音色模式",
    ),
    voice_model_choice1=dict(
        en="Voice Clone",
        zh="音色複製",
    ),
    # Speaker1 Prompt
    spk1_prompt_audio_label=dict(
        en="Speaker Prompt Audio",
        zh="說話人 參考语音",
    ),
    spk1_prompt_text_label=dict(
        en="Speaker Prompt Text",
        zh="說話人 参考文本",
    ),
    spk1_prompt_text_placeholder=dict(
        en="text of speaker prompt audio.",
        zh="說話人 参考文本",
    ),
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="對話文本輸入",
    ),
    dialogue_text_input_placeholder=dict(
        en="text...",
        zh="文本...",
    ),
    # Generate button
    generate_btn_label=dict(
        en="Generate Audio",
        zh="合成",
    ),
    # Generated audio
    generated_audio_label=dict(
        en="Generated Dialogue Audio",
        zh="合成的語音",
    ),
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text=dict(
        en='Invalid speaker 1 prompt text, should strictly follow: "xxx"',
        zh='说话人 1 参考文本不合格，格式："xxx"',
    ),
    # Warining2: invalid text for dialogue input
    warn_invalid_dialogue_text=dict(
        en='Invalid dialogue input text, should strictly follow: "xxx..."',
        zh='對话文本輸入不合格，格式："xxx..."',
    ),
    # Warining3: incomplete prompt info
    warn_incomplete_prompt=dict(
        en="Please provide prompt audio and text for both speaker 1",
        zh="請提供说话人 的參考语音與參考文本",
    ),
)

global_lang: Literal["zh", "en"] = "zh"


def i18n(key):
    global global_lang
    return _i18n_key2lang_dict[key][global_lang]


def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True

def dialogue_synthesis_function(
    target_text: str,
    voice_mode: Literal[0, 1] = 0,  # 0 means voice clone
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
):
    # Voice clone mode, check prompt info
    if voice_mode == 0:
        prompt_has_value = [
            spk1_prompt_text != "",
            spk1_prompt_audio is not None,
        ]
        if not all(prompt_has_value):
            gr.Warning(message=i18n("warn_incomplete_prompt"))
            return None

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav = None if voice_mode != 0 else spk1_prompt_audio
    prompt_text = None if voice_mode != 0 else spk1_prompt_text
    target_audio = model.generate_monologue(
        text=target_text,
        prompt_wav=prompt_wav,
        prompt_text=prompt_text,
        temperature=0.9,
        topk=30,
    )
    return (24000, target_audio.squeeze(0).cpu().numpy())


# UI rendering
def render_interface() -> gr.Blocks:
    with gr.Blocks(title="FireRedTTS-2", theme=gr.themes.Default()) as page:
        # ======================== UI ========================
        # A large title
        title_desc = gr.Markdown(value="# {}".format(i18n("title_md_desc")))
        with gr.Row():
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label="Display Language/显示语言",
                type="index",
                interactive=True,
            )
            voice_mode_choice = gr.Radio(
                choices=[i18n("voice_model_choice1")],
                value=i18n("voice_model_choice1"),
                label=i18n("voice_mode_label"),
                type="index",
                interactive=True,
            )
        with gr.Row():
            # ==== Speaker1 Prompt ====
            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk1_prompt_group:
                    spk1_prompt_audio = gr.Audio(
                        label=i18n("spk1_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )  # Audio component returns tmp audio path
                    spk1_prompt_text = gr.Textbox(
                        label=i18n("spk1_prompt_text_label"),
                        placeholder=i18n("spk1_prompt_text_placeholder"),
                        lines=3,
                    )
            # ==== Text input ====
            with gr.Column(scale=2):
                dialogue_text_input = gr.Textbox(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                    lines=18,
                )
        # Generate button
        generate_btn = gr.Button(
            value=i18n("generate_btn_label"), variant="primary", size="lg"
        )
        # Long output audio
        generate_audio = gr.Audio(
            label=i18n("generated_audio_label"),
            interactive=False,
        )

        # ======================== Action ========================
        # Language action
        def _change_component_language(lang):
            global global_lang
            global_lang = ["zh", "en"][lang]
            return [
                # title_desc
                gr.update(value="# {}".format(i18n("title_md_desc"))),
                # voice_mode_choice
                gr.update(
                    choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                    value=i18n("voice_model_choice1"),
                    label=i18n("voice_mode_label"),
                ),
                # spk1_prompt_{audio,text}
                gr.update(label=i18n("spk1_prompt_audio_label")),
                gr.update(
                    label=i18n("spk1_prompt_text_label"),
                    placeholder=i18n("spk1_prompt_text_placeholder"),
                ),
                # dialogue_text_input
                gr.update(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                ),
                # generate_btn
                gr.update(value=i18n("generate_btn_label")),
                # generate_audio
                gr.update(label=i18n("generated_audio_label")),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=[
                title_desc,
                voice_mode_choice,
                spk1_prompt_audio,
                spk1_prompt_text,
                dialogue_text_input,
                generate_btn,
                generate_audio,
            ],
        )

        # Voice clone mode action
        def _change_prompt_input_visibility(voice_mode):
            enable = voice_mode == 0
            return [gr.update(visible=enable), gr.update(visible=enable)]

        voice_mode_choice.change(
            fn=_change_prompt_input_visibility,
            inputs=[voice_mode_choice],
            outputs=[spk1_prompt_group],
        )
        generate_btn.click(
            fn=dialogue_synthesis_function,
            inputs=[
                dialogue_text_input,
                voice_mode_choice,
                spk1_prompt_text,
                spk1_prompt_audio,
            ],
            outputs=[generate_audio],
        )
    return page


# ================================================
#                   Options
# ================================================
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-dir", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Initiate model
    initiate_model(args.pretrained_dir)
    print("[INFO] FireRedTTS-2 loaded")
    # UI
    page = render_interface()
    page.launch()
