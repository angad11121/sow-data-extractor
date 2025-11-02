"""
bertopic_extraction.py

Extracts topics from SOW document SECTIONS (not entire documents).
This allows us to identify themes like "finance", "deliverables", "legal" within documents.
"""

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, ZeroShotClassification
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import glob
import os
import json
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def remove_proper_nouns(text):
    doc = nlp(text)
    # Keep everything except proper nouns (PROPN) and person names (PERSON entity)
    filtered = [token.text for token in doc if token.pos_ != "PROPN"]
    return " ".join(filtered)

# 1. Collect SOW docs and split into sections
sections = []
section_metadata = []  # Track which doc each section came from

input_paths = sorted(glob.glob("generated_data/sow_*.txt"))
for filepath in input_paths:
    doc_id = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split document into paragraphs (sections)
    # Filter out very short paragraphs (< 40 chars)
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 40]
    
    for idx, paragraph in enumerate(paragraphs):
        sections.append(paragraph)
        section_metadata.append({
            "doc_id": doc_id,
            "section_idx": idx,
            "section_text": paragraph[:200]  # First 200 chars for reference
        })

print(f"Collected {len(sections)} sections from {len(input_paths)} documents")

# 2. Configure vectorizer - more aggressive stop word filtering
custom_stop_words = [
    'project', 'ensure', 'additionally', 'shall', 'will', 'hereby',
    'agreement', 'vendor', 'client', 'sow', 'statement', 'work',
    'include', 'includes', 'including', 'provided', 'provide'
]

vectorizer_model = CountVectorizer(
    stop_words='english',
    min_df=5,  # Word must appear in at least 5 sections
    max_df=0.7,  # Ignore words in >70% of sections (too common)
    ngram_range=(1, 3)  # Include up to 3-word phrases
)

# Add custom stop words
base_stop_words = vectorizer_model.get_stop_words()
all_stop_words = list(base_stop_words) + custom_stop_words
vectorizer_model = CountVectorizer(
    stop_words=all_stop_words,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 3)
)

# 3. Use a better embedding model for semantic understanding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# # We can get these seeds from llm based preliminary extraction of topics (extract_topics_from_files.py)
# seed_topic_list = [
#     ["scope", "objectives", "deliverables", "requirements", "specifications"],
#     ["payment", "invoice", "fee", "compensation", "budget", "cost"],
#     ["legal", "liability", "warranty", "indemnification", "confidentiality", "termination"],
#     ["timeline", "schedule", "milestones", "deadline", "phases"],
#     ["compliance", "audit", "standards", "regulations", "governance"],
#     ["testing", "quality", "assurance", "validation", "acceptance"],
#     ["documentation", "manuals", "guides", "reports", "reference"],
#     ["maintenance", "support", "updates", "bug", "fixes"],
#     ["infrastructure", "cloud", "deployment", "configuration", "setup"],
#     ["security", "authentication", "authorization", "encryption", "access"]
# ]


# 4. Fit BERTopic model on SECTIONS (not full documents)
topic_labels = [
    "scope and deliverables",
    "payment and compensation", 
    "legal and compliance",
    "timeline and milestones",
    "testing and quality assurance",
    "documentation",
    "maintenance and support",
    "infrastructure and deployment",
    "security and access control"
]

# Use zero-shot classification
zeroshot_model = ZeroShotClassification(topic_labels, model="facebook/bart-large-mnli")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    representation_model=zeroshot_model,  # Use zeroshot instead
    min_topic_size=15,
    verbose=True,
    zeroshot_topic_list=topic_labels,  # Add this
    zeroshot_min_similarity=0.5  # Minimum confidence threshold
)

# Apply to sections before modeling
sections_filtered = [remove_proper_nouns(s) for s in sections]
topics, probs = topic_model.fit_transform(sections_filtered)

# 5. Reduce topics if too many (optional)
# If you get >30 topics, you can merge similar ones
num_topics = len(set(topics)) - (1 if -1 in topics else 0)
if num_topics > 25:
    print(f"\nReducing {num_topics} topics to 20...")
    topic_model.reduce_topics(sections, nr_topics=20)
    topics = topic_model.topics_
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)

# 6. Save topic summary info
topic_info = topic_model.get_topic_info()
with open("extracted_topics/bertopic_info.txt", "w", encoding="utf-8") as f:
    topic_info_no_docs = topic_info.drop(columns=["Representative_Docs"], errors="ignore")
    f.write(topic_info_no_docs.to_string(index=False))

# 7. Map sections back to documents
doc_to_topics = {}
for section_idx, (topic, metadata) in enumerate(zip(topics, section_metadata)):
    doc_id = metadata["doc_id"]
    if doc_id not in doc_to_topics:
        doc_to_topics[doc_id] = {"sections": []}
    
    doc_to_topics[doc_id]["sections"].append({
        "section_idx": metadata["section_idx"],
        "topic": int(topic),
        "section_preview": metadata["section_text"]
    })

# Aggregate: which topics appear in each document
for doc_id in doc_to_topics:
    section_topics = [s["topic"] for s in doc_to_topics[doc_id]["sections"]]
    # Get unique topics and their counts
    topic_counts = {}
    for t in section_topics:
        if t != -1:  # Exclude outliers
            topic_counts[t] = topic_counts.get(t, 0) + 1
    
    # Sort by frequency
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    doc_to_topics[doc_id]["primary_topics"] = [t[0] for t in sorted_topics[:3]]  # Top 3 topics
    doc_to_topics[doc_id]["all_topics"] = topic_counts

with open("extracted_topics/bertopic_doc_topics.json", "w", encoding="utf-8") as f:
    json.dump(doc_to_topics, f, indent=2)

# 8. Print summary statistics
print("\n=== TOPIC MODEL SUMMARY ===")
print(f"Total sections analyzed: {len(sections)}")
print(f"Total documents: {len(input_paths)}")
print(f"Number of topics found: {num_topics}")
print(f"Outlier sections (Topic -1): {sum(1 for t in topics if t == -1)}")
print(f"Outlier percentage: {(sum(1 for t in topics if t == -1) / len(sections) * 100):.1f}%")

print("\n=== TOP 10 TOPICS ===")
print(topic_info_no_docs.head(11))

print("\n=== TOPIC NAMES (for interpretation) ===")
for idx, row in topic_info.head(20).iterrows():
    if row['Topic'] != -1:
        # get_topic returns [(word, score), ...], extract just the words
        topic_words = topic_model.get_topic(row['Topic'])[:5]
        words = ', '.join([word for word, score in topic_words])
        print(f"Topic {row['Topic']}: {words}")

print("\n=== OUTPUT FILES ===")
print("Topic info: extracted_topics/bertopic_info.txt")
print("Doc→Topics mapping: extracted_topics/bertopic_doc_topics.json")
