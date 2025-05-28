import streamlit as st
from LLM_search import query_gemini_multimodal, parse_gemini_output
import os
from PIL import Image

def display_multimodal_ui(gemini_output, text_hits, image_hits):
    answer, ranked_text, ranked_images = parse_gemini_output(gemini_output)

    col1, col2, col3 = st.columns([1.2, 2.5, 1.5])

    with col1:
        st.markdown("### 📑 Relevant Articles")

        scrollable_style = """
        <style>
        .scrollable-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .card-title {
            font-weight: bold;
            color: white !important;
            text-decoration: none;
            font-size: 16px;
        }
        .card-title:hover {
            text-decoration: underline;
        }
        .card-snippet {
            color: white;
            margin-top: 6px;
        }
        .card-container {
            border: 1px solid #444;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            background-color: #222; /* темний фон для контрасту */
        }
        </style>
        """

        st.markdown(scrollable_style, unsafe_allow_html=True)
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

        for idx in ranked_text:
            hit = text_hits[idx]
            title = hit.payload.get("title", " ")
            link = hit.payload.get("url", "#")
            content = hit.payload.get("content", "")
            snippet = content[:100] + ("..." if len(content) > 100 else "")

            card_html = f"""
            <div class="card-container">
                <a href="{link}" target="_blank" class="card-title">{title}</a>
                <p class="card-snippet">{snippet}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("## 💬 Answer")
        st.write(answer)

    with col3:
        st.markdown("### 🖼️ Relevant Images")
        for i in range(0, len(ranked_images), 3):
            row_images = ranked_images[i:i+3]
            cols = st.columns(len(row_images))

            for col, idx in zip(cols, row_images):
                hit = image_hits[idx]
                title = hit.payload.get("title")
                img_path = hit.payload.get("image_path")

                with col:
                    if img_path and os.path.exists(img_path):
                        st.image(Image.open(img_path), caption=title, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {img_path}")





st.set_page_config(page_title="Multimodal RAG UI", layout="wide")
st.title("🔍 Multimodal Search Assistant (Text + Image)")
query = st.text_input("Enter your query:")

if query:
    with st.spinner("Searching and generating response..."):
        gemini_output, text_hits, image_hits = query_gemini_multimodal(query)
        display_multimodal_ui(gemini_output, text_hits, image_hits)