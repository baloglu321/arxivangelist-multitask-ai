import torch
from PIL import Image
import base64
import requests  # API çağrıları için
import io  # Resim baytlarını işlemek için
from smolagents import (
    FinalAnswerTool,
    VisitWebpageTool,
    SpeechToTextTool,
    PythonInterpreterTool,
    WebSearchTool,
    GoogleSearchTool,
    WikipediaSearchTool,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    Tool,
    CodeAgent,
    tool,
)
import requests
import math
import os
import subprocess
import whisper
import pandas as pd
import requests
import json
import re
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from smolagents.utils import encode_image_base64, make_image_url


class ImageProcess(Tool):
    name = "Image_process"
    description = "This tool use for read saved images and output is image to base64"
    inputs = {"url": {"type": "string", "description": "image path"}}
    output_type = "string"

    def forward(self, url: str) -> str:
        image = Image.open(url)

        return encode_image_base64(image)


def build_agent():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("system_prompt.txt dosyası bulunamadı, varsayılan prompt kullanılacak.")
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "
    except Exception as e:
        print(f"system_prompt.txt okunurken hata oluştu: {e}")
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "

    model = LiteLLMModel(
        model_id="ollama_chat/gemma3:27b",
        api_base="...",
        num_ctx=8192,
    )

    python_inter = PythonInterpreterTool()
    spech_to_text = SpeechToTextTool()
    visit_webpage = VisitWebpageTool()
    final_answer = FinalAnswerTool()
    image_process_tool = ImageProcess()
    tool_list = [
        python_inter,
        spech_to_text,
        visit_webpage,
        final_answer,
        image_process_tool,
    ]

    Arxivangelist = CodeAgent(
        tools=tool_list,
        model=model,
        additional_authorized_imports=[
            "requests",
            "json",
            "re",
            "subprocess",
            "pandas",
            "cv2",
            "PIL",
        ],
        add_base_tools=False,
        planning_interval=5,
    )
    return Arxivangelist


# Kullanım Örneği - ÇÖZÜM 1: Sadece metin gönder
if __name__ == "__main__":
    sample_image_path = "cca530fc-4052-43b2-b130-b30968d8aa44.png"
    image = Image.open(sample_image_path)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = (
        "..../api/generate"  # ✅ /api/chat eklendi
    )

    payload = {
        "model": "gemma3:27b",
        "prompt": "What do you see in this picture?",
        "images": [image_base64],
        "stream": False,
    }

    try:
        response = requests.post(
            url, headers={"Content-Type": "application/json"}, json=payload
        )
        response.raise_for_status()  # Hatalı HTTP durum kodu varsa Exception atar

        data = response.json()

        if "response" in data:
            print(data["response"])
        else:
            print("Yanıtta 'message' anahtarı yok. Gelen veri:", data)

    except requests.exceptions.RequestException as e:
        print("İstek sırasında hata oluştu:", e)
    except Exception as e:
        print("Beklenmeyen bir hata oluştu:", e)

    # Çözüm 1: Sadece metin olarak soru sor

    Arxivangelist = build_agent()
    answer = Arxivangelist.run("Whatsup")
    print(answer)
