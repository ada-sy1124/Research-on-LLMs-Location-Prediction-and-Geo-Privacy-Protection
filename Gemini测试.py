from PIL import Image
from google import genai
from google.genai import types

# 1. 必填：替换为你的真实 API KEY
API_KEY = "" 

# 2. 参数设置 (我将模型修正为了真实存在且能力极强的最新版本)
MODEL = "gemini-3-flash-preview"  

IMAGE_PATH = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images\4751579_8241809_bbox_masked.jpg"
# IMAGE_PATH = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images\sample_0_part2.png"



PROMPT="""You are an advanced geolocation model. Your objective is to analyze the image and estimate its geographic coordinates using only visible evidence, and isolate the precise physical objects that leak this location.

You must output your response in EXACTLY four XML blocks: <Chain of Thought>, <Location>, <CoT_Keywords>, and <Anchors>. Do not output any text outside of these blocks.

1. <Chain of Thought>
Perform a step-by-step geographic deduction. Think carefully and logically to predict the location.
Explain the CAUSAL RELATIONSHIP between what you see and the location (e.g., "The presence of X implies Y because Z"). You are allowed to use semantic reasoning, read text, and identify functional objects (e.g., "fishing boats", "restaurant signs") to maximize geolocation accuracy.

2. <Location>
Output only the latitude and longitude, separated by a comma.

3. <CoT_Keywords>
RAW OBJECT EXTRACTION:
Explicitly list the semantic physical objects you relied on in your <Chain of Thought>. You must simply copy the original concepts here (e.g., "Korean commercial neon signs, fishing boats, tall buildings"). Separate them with commas. Do not add any new objects that were not mentioned in your deduction.

4. <Anchors>
CRITICAL COGNITIVE ABSTRACTION & ALIGNMENT TASK (READ CAREFULLY):
Step 1: Look at the exact physical objects extracted in your <CoT_Keywords>. Do not look back at the image to list new nouns.
Step 2: Strictly translate these semantic objects into "Basic-Level Categories" (Entry-level physical base nouns), so that a downstream zero-shot segmentation model can easily draw physical bounding boxes around them.

You must formulate your translated Anchors using the following logic:
- RULE 1: Output ONLY nouns, do not include any extra output.
- RULE 2: NO SUBORDINATE CATEGORIES (Prevent Specificity). Do not use highly specific sub-types or derived nouns. 
  * Example: Items like "SUV", "pickup truck", or "yellow taxi" should be translated to "cars".
  * Example: Items like "neon sign" or "restaurant marquee" should be translated to "signboards".
- RULE 3: NO SUPERORDINATE CATEGORIES (Prevent Abstract Macros). The noun must be a thing with a physical form. Do not use broad functional abstractions that lack a unified shape.
  * Example: Do NOT output "infrastructure"; you must output specific base nouns like "traffic lights" or "utility poles".
  * Example: Do NOT output "vehicles"; you must output "cars", "airplanes", or "trains".
- RULE 4: Output the same noun only once. In the case of multiple targets, simply output it in plural form, separated by commas.

EXAMPLE TRANSLATION LOGIC:
- If CoT mentions "Korean commercial neon signs", Anchors translates to: "signboards"
- If CoT mentions "white fishing boats", Anchors translates to: "boats"
- If CoT mentions "dense overhead high-voltage wires", Anchors translates to: "utility poles"
- If CoT mentions "Ford police SUVs", Anchors translates to: "cars"
"""


def main():
    # 尝试打开图片，顺便做个基础的防呆检查
    try:
        image = Image.open(IMAGE_PATH)
    except FileNotFoundError:
        print(f"❌ 找不到图片，请检查路径是否拼写正确:\n{IMAGE_PATH}")
        return

    print("⏳ 正在连接 Gemini API 并分析图片，请稍候...")
    
    # 初始化客户端
    client = genai.Client(api_key=API_KEY)

    # 发送请求
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[image, PROMPT.strip()],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        print("\n========== ✅ 模型输出 ==========\n")
        print(response.text)
        print("\n=================================")
        
    except Exception as e:
        print(f"\n❌ API 请求失败，错误信息: {e}")

if __name__ == "__main__":
    main()


# python ./Gemini测试.py

# 47.51579, 8.241809
# 47.50413, 8.24845
# 46.1200, 14.8150 
# 36.3894, 139.0634 矩形