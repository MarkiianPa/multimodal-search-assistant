import google.generativeai as genai
import re
from PIL import Image
from dotenv import load_dotenv
import os
import open_clip
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models as rest
from PIL import Image
import torch
import os

load_dotenv('data.env')

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


model = genai.GenerativeModel("gemini-2.0-flash")

text_model = SentenceTransformer("intfloat/e5-base")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')


qdrant = QdrantClient("http://localhost", port=6333)


def get_query_vector(query: str):
    query = f"query: {query}"
    return text_model.encode(query, normalize_embeddings=True)


def get_query_vector_clip(query: str):

    tokenized = tokenizer([query])


    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features[0].cpu().numpy()
def search_text(query_vector, top_k=10):
    return qdrant.query_points(
        collection_name="articles_collection",
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=rest.Filter(
            must=[rest.FieldCondition(key="type", match=rest.MatchValue(value="text"))]
        ),
        with_payload=True
    ).points


def search_images(query_vector, top_k=10):
    return qdrant.query_points(
        collection_name="articles_collection",
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=rest.Filter(
            must=[rest.FieldCondition(key="type", match=rest.MatchValue(value="image"))]
        ),
        with_payload=True
    ).points


def load_image(path):
    return Image.open(path)


def build_multimodal_gemini_prompt(query, text_hits, image_hits):
    prompt = f"""
You are a helpful multimodal assistant.

The user is searching for: "{query}"

You are provided with several candidate results retrieved from a database: 
- text snippets (articles) with titles
- images with titles.

Your task:

1. Analyze the user's query and answer it **based on the retrieved content** (both texts and images). Do not make assumptions beyond the provided materials.

2. Rank **all the candidate results** (texts and images) **by their relevance to the user's query**. 
   **rank all articles and images**
   **do not make duplicate copies**
   For each ranked item, provide:
   - its type (Text/Image)
   - its number (as given)


### Format:

Answer:
<your answer to the user's query>

Ranked Text Results:
1. [Text #N] — Title #N
2. [Text #N] — Title #N
3. ...

Ranked Image Results:
1. [Image #N] — Title #N
2. [Image #N] — Title #N
3. ...
"""
    inputs = [prompt.strip()]
    for i, hit in enumerate(text_hits):
        title = hit.payload.get("title", "No title")
        text = hit.payload.get("content", "No content")
        combined = f"Text #{i + 1}:\nTitle: {title}\nContent: {text}"
        inputs.append(combined)

    for i, hit in enumerate(image_hits):
        caption = hit.payload.get("title", "No title")
        image_path = hit.payload.get("image_path")
        if image_path and os.path.exists(image_path):
            img = load_image(image_path)
            inputs.append(f"Image #{i + 1}:\nTitle: {caption}\nImage: {img}")
        else:
            inputs.append(f"Image #{i + 1}: [Missing image at {image_path}]")
    return inputs


def query_gemini_multimodal(query):
    q_vec_text = get_query_vector(query)
    q_vec_image = get_query_vector_clip(query)
    text_hits = search_text(q_vec_text)
    image_hits = search_images(q_vec_image)

    gemini_input = build_multimodal_gemini_prompt(query, text_hits, image_hits)
    response = model.generate_content(gemini_input, stream=False)
    return response.text, text_hits, image_hits


def parse_gemini_output(gemini_output):
    answer_match = re.search(r"Answer:\n(.+?)\n\n", gemini_output, re.DOTALL)
    text_matches = re.findall(r"\d+\.\s+\[Text\s+#(\d+)]\s+—\s+(.*)", gemini_output)
    image_matches = re.findall(r"\d+\.\s+\[Image\s+#(\d+)]\s+—\s+(.*)", gemini_output)

    answer = answer_match.group(1).strip() if answer_match else "[No answer found]"
    ranked_text = [(int(idx) - 1) for idx, title in text_matches]
    ranked_images = [(int(idx) - 1) for idx, title in image_matches]

    return answer, ranked_text, ranked_images