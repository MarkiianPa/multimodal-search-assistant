import torch
import open_clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import qdrant_client
from qdrant_client.http import models as rest
import io
from tqdm import tqdm

text_model = SentenceTransformer('intfloat/e5-base')

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
clip_model.eval()

qdrant = qdrant_client.QdrantClient(url="http://localhost:6333")


def get_text_embedding(title, content):
    text = title + " " + content
    emb = text_model.encode(text, convert_to_tensor=True)
    return emb.cpu().numpy()


def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        return None
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0]


def upsert_to_qdrant(df, collection_name="articles_collection"):
    vector_size_text = 768
    vector_size_image = 768

    if not qdrant.collection_exists(collection_name):
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size_text, distance=rest.Distance.COSINE)
        )

    points = []
    point_id = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Uploading to Qdrant"):
        title = row['title']
        content = row['content']

        text_emb = get_text_embedding(title, content)
        points.append(rest.PointStruct(
            id=point_id,
            vector=text_emb.tolist(),
            payload={"url": row['url'], "title": title, "content": content, "type": "text"}
        ))
        point_id += 1

        if isinstance(row['media_urls'], list) and len(row['media_urls']) > 0:
            for media_path in row['media_urls']:

                if os.path.exists(media_path):
                    img_emb = get_image_embedding(media_path)

                    if img_emb is not None:
                        points.append(rest.PointStruct(
                            id=point_id,
                            vector=img_emb.tolist(),
                            payload={
                                "title": title,
                                "type": "image",
                                "image_path": media_path
                            }
                        ))
                        point_id += 1

        if len(points) >= 100:
            qdrant.upsert(collection_name=collection_name, points=points)
            points = []

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv('data/articles_with_local_images.csv', converters={'media_urls': ast.literal_eval})
    upsert_to_qdrant(df)
    print('Data ingested')