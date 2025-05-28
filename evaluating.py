from LLM_search import get_query_vector, get_query_vector_clip, search_text, search_images, model, load_image
import argparse
import json
import re
import os

def build_multimodal_gemini_prompt(query, text_hits, image_hits):
    prompt = f"""
You are a helpful multimodal assistant.

The user is searching for: "{query}"

You are provided with several candidate results retrieved from a database: 
- text snippets (articles) with titles
- images with titles.

Your task:

1. Rank **all the candidate results** (texts and images) **by their relevance to the user's query**. 
   **Rank all articles and all images together by relevance, but report them in separate lists**.

2. For each ranked item, assign a **relevance score**  **0 (not relevant)** or **1 (relevant)** based on how well the item matches the query.

3. Do not repeat or duplicate results.

For each ranked item, provide:
- its type (Text/Image)
- its number (as given)
- its title
- its relevance score (0 or 1)

### Format:

Answer:
<your answer to the user's query>

Ranked Text Results:
1. [Text #N] — Title: "<Title>" — Score: <relevance_score>
2. ...

Ranked Image Results:
1. [Image #N] — Title: "<Title>" — Score: <relevance_score>
2. ...
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

def parse_ranked_results(model_output):
    """
    Parses the model output to extract text and image results with their scores.
    Returns two lists: texts and images, each containing dicts with keys:
    {'rank': int, 'id': int, 'title': str, 'score': float}
    """
    text_pattern = re.compile(
        r"\[\s*Text\s*#(\d+)\s*\]\s*—\s*Title:\s*\"(.*?)\"\s*—\s*Score:\s*(\d+(?:\.\d+)?)"
    )
    image_pattern = re.compile(
        r"\[\s*Image\s*#(\d+)\s*\]\s*—\s*Title:\s*\"(.*?)\"\s*—\s*Score:\s*(\d+(?:\.\d+)?)"
    )

    texts = [
        {"id": int(i), "title": title, "score": float(score)}
        for i, title, score in text_pattern.findall(model_output)
    ]
    images = [
        {"id": int(i), "title": title, "score": float(score)}
        for i, title, score in image_pattern.findall(model_output)
    ]

    return texts, images


def evaluate_retrieval_metrics(texts, images, k=3):
    """
    Computes Precision@K, Recall@K, and F1@K for both texts and images.

    Assumes that items with score >= relevance_threshold are relevant.
    """

    def compute_metrics(top_items, total_items, k):
        relevant_total = sum(1 for item in total_items if item["score"] == 1)
        relevant_at_k = sum(1 for item in top_items[:k] if item["score"] == 1)

        precision = relevant_at_k / k if k > 0 else 0.0
        recall = relevant_at_k / relevant_total if relevant_total > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision@k": round(precision, 4),
            "recall@k": round(recall, 4),
            "f1@k": round(f1, 4),
            "relevant@k": relevant_at_k,
            "total_relevant": relevant_total
        }

    texts_sorted = sorted(texts, key=lambda x: x["score"], reverse=True)
    images_sorted = sorted(images, key=lambda x: x["score"], reverse=True)

    text_metrics = compute_metrics(texts_sorted[:k], texts, k)
    image_metrics = compute_metrics(images_sorted[:k], images, k)

    return {
        "text_metrics": text_metrics,
        "image_metrics": image_metrics
    }


def evaluate_multiple_queries(queries, k=3):
    """
    Evaluate retrieval performance across multiple queries.

    Returns per-query results and average metrics across all queries.
    """
    all_text_precisions, all_text_recalls, all_text_f1s = [], [], []
    all_image_precisions, all_image_recalls, all_image_f1s = [], [], []

    per_query_metrics = []

    for query in queries:
        output, text_hits, image_hits = query_gemini_multimodal(query)
        texts, images = parse_ranked_results(output)

        metrics = evaluate_retrieval_metrics(texts, images, k=k)
        per_query_metrics.append({
            "query": query,
            "text_metrics": metrics["text_metrics"],
            "image_metrics": metrics["image_metrics"]
        })

        # Accumulate
        all_text_precisions.append(metrics["text_metrics"]["precision@k"])
        all_text_recalls.append(metrics["text_metrics"]["recall@k"])
        all_text_f1s.append(metrics["text_metrics"]["f1@k"])

        all_image_precisions.append(metrics["image_metrics"]["precision@k"])
        all_image_recalls.append(metrics["image_metrics"]["recall@k"])
        all_image_f1s.append(metrics["image_metrics"]["f1@k"])

    # Compute macro-average
    avg_metrics = {
        "text": {
            "avg_precision@k": round(sum(all_text_precisions) / len(all_text_precisions), 4),
            "avg_recall@k": round(sum(all_text_recalls) / len(all_text_recalls), 4),
            "avg_f1@k": round(sum(all_text_f1s) / len(all_text_f1s), 4),
        },
        "image": {
            "avg_precision@k": round(sum(all_image_precisions) / len(all_image_precisions), 4),
            "avg_recall@k": round(sum(all_image_recalls) / len(all_image_recalls), 4),
            "avg_f1@k": round(sum(all_image_f1s) / len(all_image_f1s), 4),
        },
    }

    return {
        "per_query": per_query_metrics,
        "avg_metrics": avg_metrics
    }


def output_metrics(results):
    print("\n📊 Average Metrics across all queries:")
    print(f"Text - Precision@3: {results['avg_metrics']['text']['avg_precision@k']}")
    print(f"Text - Recall@3:    {results['avg_metrics']['text']['avg_recall@k']}")
    print(f"Text - F1@3:        {results['avg_metrics']['text']['avg_f1@k']}")
    print('\n')
    print(f"Image - Precision@3: {results['avg_metrics']['image']['avg_precision@k']}")
    print(f"Image - Recall@3:    {results['avg_metrics']['image']['avg_recall@k']}")
    print(f"Image - F1@3:        {results['avg_metrics']['image']['avg_f1@k']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple queries.")
    parser.add_argument(
        '--queries',
        type=str,
        required=True,
        help='JSON-encoded list of queries (use double quotes around the list)'
    )
    args = parser.parse_args()

    queries = json.loads(args.queries)

    results = evaluate_multiple_queries(queries, k=3)
    output_metrics(results)
