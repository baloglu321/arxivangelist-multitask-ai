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
import base64
import io

os.environ["WEATHER_API"] = "..."
os.environ["SERPER_API_KEY"] = "..."
ollama_server = "..."
model_id = "gemma3:27b"


def get_question():
    API_URL = "https://agents-course-unit4-scoring.hf.space/random-question"
    response = requests.get(API_URL).json()

    question = response.get("question")
    return question, response


class CustomError(Exception):
    pass


class SearchTool(Tool):
    name = "Search_Tool"
    description = (
        "Use this tool for all general-purpose web searches. "
        "It replaces all other web search tools. "
        "If the user wants to search the internet, always prefer this tool.")
    inputs = {
        "query": {
            "type": "string",
            "description": "The info the user wants"
        }
    }
    output_type = "string"

    def forward(self, query: str):
        self.tool = DuckDuckGoSearchTool()
        results = self.tool.forward(query)
        return results

    def __call__(self, query: str):
        return self.forward(query)


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str):

        url = f"https://api.weatherstack.com/current?access_key={WEATHER_API}"
        querystring = {"query": location}
        response = requests.get(url, params=querystring)
        data = response.json()
        city = data["location"]["name"]
        country = data["location"]["country"]
        temperature = data["current"]["temperature"]
        weather_description = data["current"]["weather_descriptions"][0]
        return f"Weather in {location}: {weather_description}, {str(temperature)}°C"


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    inputs = {
        "a": {
            "type": "number",
            "description": "first number"
        },
        "b": {
            "type": "number",
            "description": "second number"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a * b

    def __call__(self, a: float, b: float) -> float:
        return self.forward(a, b)


class AddTool(Tool):
    name = "add_tool"
    description = "add two numbers"
    inputs = {
        "a": {
            "type": "number",
            "description": "first number"
        },
        "b": {
            "type": "number",
            "description": "second number"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a + b

    def __call__(self, a: float, b: float) -> float:
        return self.forward(a, b)


class SubtractTool(Tool):
    name = "subtract_tool"
    description = "subtract two numbers"
    inputs = {
        "a": {
            "type": "number",
            "description": "first number"
        },
        "b": {
            "type": "number",
            "description": "second number"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a - b

    def __call__(self, a: float, b: float) -> float:
        return self.forward(a, b)


class DivideTool(Tool):
    name = "divide_tool"
    description = "divide two numbers"
    inputs = {
        "a": {
            "type": "number",
            "description": "first number"
        },
        "b": {
            "type": "number",
            "description": "second number"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def __call__(self, a: float, b: float) -> float:
        return self.forward(a, b)


class PowerTool(Tool):
    name = "power_tool"
    description = "raise a number to a power"
    inputs = {
        "a": {
            "type": "number",
            "description": "number"
        },
        "b": {
            "type": "number",
            "description": "power"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float) -> float:
        return a**b

    def __call__(self, a: float, b: float) -> float:
        return self.forward(a, b)


class SquareRootTool(Tool):
    name = "square_root_tool"
    description = "calculate the square root of a number"
    inputs = {"a": {"type": "number", "description": "number"}}
    output_type = "number"

    def forward(self, a: float):
        return a**0.5

    def __call__(self, a: float):
        return self.forward(a)


class LogarithmTool(Tool):
    name = "logarithm_tool"
    description = "calculate the logarithm of a number"

    inputs = {"a": {"type": "number", "description": "number"}}
    output_type = "number"

    def forward(self, a: float):
        return math.log(a)

    def __call__(self, a: float):
        return self.forward(a)


class NaturalLogarithmTool(Tool):
    name = "natural_logarithm_tool"
    description = "calculate the natural logarithm of a number"
    inputs = {"a": {"type": "number", "description": "number"}}
    output_type = "number"

    def forward(self, a: float):
        return math.log(a)

    def __call__(self, a: float):
        return self.forward(a)


class ModulusTool(Tool):
    name = "modulus_tool"
    description = "calculate the modulus of two numbers"
    inputs = {
        "a": {
            "type": "number",
            "description": "first number"
        },
        "b": {
            "type": "number",
            "description": "second number"
        },
    }
    output_type = "number"

    def forward(self, a: float, b: float):
        return a % b

    def __call__(self, a: float, b: float):
        return self.forward(a, b)


class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Search Wikipedia for a query and return up to 2 results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        from langchain_community.document_loaders import WikipediaLoader

        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        return "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = "Search Arxiv for a query and return up to 3 results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        from langchain_community.document_loaders import ArxivLoader

        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        return "\n\n---\n\n".join([
            f'<Document source="{doc.metadata.get("source", "unknown")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])


def download_audio_from_youtube(url, output_path="audio.mp3"):
    subprocess.run([
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        output_path,
        url,
    ])


def download_video_from_youtube(url, output_path="video.mp4"):
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo+bestaudio", "-o", output_path, url],
        capture_output=True,
        text=True,
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")


def transcribe_audio_whisper(audio_path):
    model = whisper.load_model(
        "base")  # 'tiny', 'base', 'small', 'medium', 'large'
    result = model.transcribe(audio_path)
    return result["text"]


def youtube_transcript_tool(url):
    audio_path = "audio.mp3"
    download_audio_from_youtube(url, audio_path)
    transcript = transcribe_audio_whisper(audio_path)
    os.remove(audio_path)
    return transcript


class TranscriberTool(Tool):
    name = "mp3_transcript"
    description = "Extracts transcript from any voice file using Whisper"
    inputs = {
        "path": {
            "type": "string",
            "description": "Voice path to be transcribed."
        }
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        return transcribe_audio_whisper(path)


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = "Extracts transcript from a YouTube video using Whisper"
    inputs = {
        "url": {
            "type": "string",
            "description": "Video link to be transcribed."
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        return youtube_transcript_tool(url)


class DataProcessingTool(Tool):
    """Veri işleme ve analiz için genelleştirilmiş tool"""

    name = "data_processing"
    description = "Process data: parse tables, extract numbers, format text, statistical calculations"
    inputs = {
        "data": {
            "type": "string",
            "description": "Input data to process"
        },
        "operation": {
            "type":
            "string",
            "description":
            "Operation: 'extract_numbers', 'parse_table', 'calculate', 'format_text', 'find_pattern'",
        },
        "parameters": {
            "type": "string",
            "description": "Additional parameters as JSON string (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, data: str, operation: str, parameters: str = None):
        try:
            params = json.loads(parameters) if parameters else {}

            if operation == "extract_numbers":
                numbers = re.findall(r"-?\d+(?:\.\d+)?", data)
                return f"Extracted numbers: {numbers}"

            elif operation == "parse_table":
                # Basit tablo parsing
                lines = data.strip().split("\n")
                table_data = []
                for line in lines:
                    if "|" in line:
                        row = [
                            cell.strip() for cell in line.split("|")
                            if cell.strip()
                        ]
                        table_data.append(row)
                return f"Parsed table: {table_data}"

            elif operation == "calculate":
                # Matematiksel ifadeleri güvenli şekilde hesapla
                try:
                    # Sadece temel matematiksel operasyonları destekle
                    allowed_chars = set("0123456789+-*/.() ")
                    if all(c in allowed_chars for c in data):
                        result = eval(data)
                        return f"Calculation result: {result}"
                    else:
                        return "Invalid mathematical expression"
                except:
                    return "Could not calculate expression"

            elif operation == "format_text":
                format_type = params.get("format", "clean")
                if format_type == "clean":
                    # Metni temizle
                    cleaned = re.sub(r"\s+", " ", data.strip())
                    return cleaned
                elif format_type == "upper":
                    return data.upper()
                elif format_type == "lower":
                    return data.lower()
                elif format_type == "title":
                    return data.title()

            elif operation == "find_pattern":
                pattern = params.get("pattern", "")
                if pattern:
                    matches = re.findall(pattern, data)
                    return f"Pattern matches: {matches}"
                else:
                    return "No pattern specified"

        except Exception as e:
            return f"Error in data processing: {str(e)}"


class WebScrapingTool(Tool):
    """Gelişmiş web scraping ve analiz tool'u"""

    name = "web_scraping"
    description = "Advanced web scraping: extract specific data from web pages, parse tables, find links"
    inputs = {
        "url": {
            "type": "string",
            "description": "URL to scrape"
        },
        "target": {
            "type":
            "string",
            "description":
            "Target: 'tables', 'links', 'text', 'images', 'specific_element'",
        },
        "selector": {
            "type": "string",
            "description":
            "CSS selector or XPath for specific elements (optional)",
            "nullable": True,
        },
        "filters": {
            "type": "string",
            "description": "Filters as JSON string (optional)",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self,
                url: str,
                target: str,
                selector: str = None,
                filters: str = None):
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Default values
            selector = selector or ""
            filters = filters or "{}"
            filter_dict = json.loads(filters)

            if target == "tables":
                tables = soup.find_all("table")
                table_data = []
                for i, table in enumerate(tables[:3]):  # İlk 3 tablo
                    rows = []
                    for row in table.find_all("tr"):
                        cells = [
                            cell.get_text().strip()
                            for cell in row.find_all(["td", "th"])
                        ]
                        if cells:
                            rows.append(cells)
                    table_data.append(f"Table {i+1}: {rows}")
                return "\n".join(table_data)

            elif target == "links":
                links = soup.find_all("a", href=True)
                link_list = [(link.get_text().strip(), link["href"])
                             for link in links[:20]]
                return f"Links found: {link_list}"

            elif target == "text":
                # Ana metni çıkar
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines
                          for phrase in line.split("  "))
                text = " ".join(chunk for chunk in chunks if chunk)
                return text[:2000]  # İlk 2000 karakter

            elif target == "specific_element" and selector:
                elements = soup.select(selector)
                results = [elem.get_text().strip() for elem in elements[:10]]
                return f"Selected elements: {results}"

            elif target == "images":
                images = soup.find_all("img", src=True)
                img_list = [(img.get("alt", "No alt"), img["src"])
                            for img in images[:10]]
                return f"Images found: {img_list}"

        except Exception as e:
            return f"Error in web scraping: {str(e)}"


def caption_image(image_path: str, prompt: str) -> str:
    global ollama_server
    global model_id

    image = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = ollama_server + "/api/generate"

    payload = {
        "model": model_id,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
    }
    response = requests.post(url,
                             headers={"Content-Type": "application/json"},
                             json=payload)
    response.raise_for_status()  # Hatalı HTTP durum kodu varsa Exception atar

    data = response.json()

    if "response" in data:
        return data["response"]
    else:
        return "Image not recognized"


class ImageCaptionerTool(Tool):
    name = "image_captioner"
    description = (
        "Generates a detailed natural language description of the given image using a multimodal large language model "
        "via an Ollama server. Accepts a local image file (e.g., JPG or PNG) and a textual prompt describing "
        "what to look for in the image. The image is encoded in base64 and sent to the model for visual understanding."
    )

    inputs = {
        "path": {
            "type":
            "string",
            "description":
            "The local file path of the image to describe (e.g., JPG or PNG).",
        },
        "text": {
            "type":
            "string",
            "description":
            "What are you looking for in the image? (e.g. 'How many people are wearing helmets?')",
        },
    }
    output_type = "string"

    def forward(self, path: str, text: str) -> str:
        return caption_image(path, prompt=text)


class FileDownloadTool(Tool):
    name = "File_Download_Tool"
    description = (
        "Downloads a file from a Hugging Face-hosted URL using the given task_id.\n\n"
        "Supported file types and behaviors:\n"
        "- If .mp3 → Downloads the audio file and returns its saved path.\n"
        "- If .xlsx → Parses the Excel file and returns a readable text \n"
        "- If .json → Parses and returns the full JSON content in readable format.\n"
        "- If .jpg/.jpeg/.png/.bmp → Downloads the image file and returns its saved path.\n"
        "- For all other file types → Deletes the file and returns an unsupported format message."
    )

    inputs = {
        "task_id": {
            "type":
            "string",
            "description":
            ("The task_id used to construct the Hugging Face file download URL. "
             "This ID corresponds to a specific file uploaded to the course environment."
             ),
        }
    }
    output_type = "string"

    def forward(self, task_id: str) -> str:
        url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
        response = requests.get(url)

        content_disposition = response.headers.get("content-disposition", "")
        if "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            return "Unable to find a supported file type."

        file_path = os.path.join(".", filename)
        with open(file_path, "wb") as f:
            f.write(response.content)

        if filename.endswith(".mp3"):
            answer = f"The MP3 file was downloaded successfully. Saved at: {file_path}"
            return answer
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file_path)

            # İlk birkaç satırı düzgün formatta yazalım (çok büyükse tümünü yazmak istemeyebiliriz)
            df_preview = df.to_string(index=False)

            return (
                f"The file '{filename}' has been downloaded and read as an Excel spreadsheet.\n"
                f"Here is a preview of its contents:\n\n{df_preview}")

        elif filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"The file '{filename}' has been downloaded and its content is as follows:\n{json.dumps(data, indent=2)}"

        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".PNG",
                                ".JPG", ".JPEG", ".BMP")):
            answer = (
                f"The image file was downloaded successfully. Saved at: {file_path}"
            )
            return answer
        else:
            os.remove(file_path)  # gereksiz dosyayı sil
            return "The downloaded file is not in a supported format."

    def __call__(self, task_id: str) -> str:
        return self.forward(task_id)

    def __call__(self, task_id: str) -> str:
        return self.forward(task_id)


def download_video_from_youtube(url, output_path="video.mp4"):
    result = subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]",
            "--merge-output-format",
            "mp4",  # video+ses -> mp4
            "-o",
            output_path,
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed, {output_path} not found.")

    return output_path


class YouTubeDownloadTool(Tool):
    name = "youtube_video_download"
    description = (
        "This tool downloads a YouTube video from the provided URL using yt-dlp. "
        "After downloading, it returns the local path to the saved video file."
    )
    inputs = {
        "url": {
            "type":
            "string",
            "description":
            "The YouTube video URL to download. Returns the path of the downloaded video file.",
        },
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        video_path = download_video_from_youtube(url)
        answer = f"The video file was downloaded successfully. Saved at: {video_path}"
        return answer


def build_agent():
    global ollama_server
    global model_id
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print(
            "system_prompt.txt dosyası bulunamadı, varsayılan prompt kullanılacak."
        )
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "
    except Exception as e:
        print(f"system_prompt.txt okunurken hata oluştu: {e}")
        system_prompt = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with 'FINAL ANSWER: ', then follows with the answer. "

    model = LiteLLMModel(
        # model_id="ollama_chat/qwen2.5-coder:14b-instruct-fp16", #20/3
        # model_id="ollama_chat/deepseek-coder:33b-instruct",  #0/20 ~6000sec
        # provider="together",
        model_id=f"ollama_chat/{model_id}",
        api_base=ollama_server,  # Default Ollama local server
        num_ctx=8192,
        system_prompt=system_prompt,
    )

    audio_transcribe = TranscriberTool()
    youtube_video_downloader = YouTubeDownloadTool()
    youtube_transcribe = YouTubeTranscriptTool()
    image_caption = ImageCaptionerTool()
    web_search = GoogleSearchTool("serper")
    python_inter = PythonInterpreterTool()
    spech_to_text = SpeechToTextTool()
    visit_webpage = VisitWebpageTool()
    final_answer = FinalAnswerTool()
    weather_info_tool = WeatherInfoTool()
    wiki_search_tool = WikiSearchTool()
    arxiv_search_tool = ArxivSearchTool()
    data_tool = DataProcessingTool()
    web_tool = WebScrapingTool()
    file_download = FileDownloadTool()
    multiply_tool = MultiplyTool()
    add_tool = AddTool()
    subtract_tool = SubtractTool()
    divide_tool = DivideTool()
    power_tool = PowerTool()
    square_root_tool = SquareRootTool()
    logarithm_tool = LogarithmTool()
    natural_logarithm_tool = NaturalLogarithmTool()
    modulus_tool = ModulusTool()
    tool_list = [
        audio_transcribe,
        image_caption,
        youtube_video_downloader,
        file_download,
        data_tool,
        web_tool,
        weather_info_tool,
        youtube_transcribe,
        web_search,
        python_inter,
        spech_to_text,
        visit_webpage,
        final_answer,
        wiki_search_tool,
        arxiv_search_tool,
        multiply_tool,
        add_tool,
        subtract_tool,
        divide_tool,
        power_tool,
        square_root_tool,
        logarithm_tool,
        natural_logarithm_tool,
        modulus_tool,
    ]

    Arxivangelist = CodeAgent(
        tools=tool_list,
        model=model,
        # managed_agents=[video_agent],
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


def tool_test():

    tool_tests = {
        "multiply": lambda tool: tool(a=5, b=3),  # 15
        "add_tool": lambda tool: tool(a=7, b=2),  # 9
        "subtract_tool": lambda tool: tool(a=10, b=4),  # 6
        "divide_tool": lambda tool: tool(a=8, b=2),  # 4
        "power_tool": lambda tool: tool(a=3, b=3),  # 27
        "square_root_tool": lambda tool: tool(a=16),  # 4
        "logarithm_tool": lambda tool: tool(a=10),  # ~2.302
        "natural_logarithm_tool": lambda tool: tool(a=10),  # ~2.302
        "modulus_tool": lambda tool: tool(a=10, b=3),  # 1
        "wiki_search": lambda tool: tool.forward("Python programming"),
        "arxiv_search": lambda tool: tool.forward("transformers in NLP"),
        "Search_Tool": lambda tool: tool.forward("Latest AI news"),
        "weather_info": lambda tool: tool.forward("Istanbul"),
    }
    weather_info_tool = WeatherInfoTool()
    wiki_search_tool = WikiSearchTool()
    arxiv_search_tool = ArxivSearchTool()
    tool_list = [weather_info_tool, wiki_search_tool, arxiv_search_tool]

    for tool in tool_list:
        try:
            test_fn = tool_tests.get(tool.name)
            if not test_fn:
                print(f"⚠️ No test case defined for tool: {tool.name}")
                continue
            print(f"\n🔧 Running {tool.name}...")
            result = test_fn(tool)
            print(f"✅ Result: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error in tool {tool.name}: {e}")
            print("-" * 50)


if __name__ == "__main__":
    # tool_test()
    Arxivangelist = build_agent()
    question, response = get_question()
    answer = Arxivangelist.run(question)
    print(answer)
