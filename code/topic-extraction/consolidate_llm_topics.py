"""
consolidate_llm_topics.py

Consolidates the LLM-extracted topics from sow_topics.json into 
a canonical list with consistent naming and IDs.
"""

import json
import asyncio
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import sys
import os
from code.helpers.llm import SimpleLLMCaller



# ============================================================================
# CONFIGURATION
# ============================================================================
LLM_MODEL = "gpt-4.1"  # Model to use for generating canonical topic names
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better sentence transformer for topic clustering


def load_and_analyze_topics(json_path: str):
    """Load topics and compute statistics."""
    with open(json_path, 'r') as f:
        sow_topics = json.load(f)
    
    all_topics = []
    for doc_id, topics in sow_topics.items():
        all_topics.extend(topics)
    
    topic_freq = Counter(all_topics)
    
    print(f"Loaded {len(sow_topics)} documents with {len(all_topics)} topic mentions ({len(topic_freq)} unique)")
    print(f"Average topics per document: {len(all_topics) / len(sow_topics):.1f}")
    
    print(f"\nTop 20 most common topics:")
    for topic, count in topic_freq.most_common(20):
        coverage = count / len(sow_topics) * 100
        print(f"  {topic:40s}: {count:4d} docs ({coverage:5.1f}%)")
    
    return sow_topics, topic_freq


def cluster_similar_topics(topic_freq: Counter, similarity_threshold: float = 0.7):
    """Cluster semantically similar topic names using embeddings."""
    unique_topics = list(topic_freq.keys())
    
    print(f"\nClustering {len(unique_topics)} unique topics (threshold: {similarity_threshold})")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(unique_topics)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    clusters = {}
    for topic, label in zip(unique_topics, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((topic, topic_freq[topic]))
    
    for label in clusters:
        clusters[label].sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(clusters)} topic clusters")
    
    return clusters, embeddings, unique_topics


async def generate_canonical_names(clusters: dict):
    """Use LLM to generate canonical names for each cluster."""
    llm_caller = SimpleLLMCaller(max_retries=3, semaphore=10)
    
    print(f"\nGenerating canonical names using {LLM_MODEL}")
    
    canonical_topics = {}
    sem = asyncio.Semaphore(10)
    
    async def process_cluster(cluster_id, topic_variants):
        async with sem:
            # Prepare topic list with frequencies
            topic_list = "\n".join([
                f"  - \"{topic}\" (appears in {count} documents)"
                for topic, count in topic_variants[:10]  # Top 10 variants
            ])
            
            prompt = f"""You are analyzing Statement of Work (SOW) document topics.

The following topic names all refer to the SAME semantic concept:

{topic_list}

Task:
1. Choose the BEST canonical name (2-4 words, clear and professional)
2. Determine if this is a MAIN topic or a SUBTOPIC
3. If subtopic, identify the parent topic
4. List 3-5 common subtopics that would fall under this category

Return ONLY valid JSON (no markdown formatting):
{{
  "canonical_name": "...",
  "type": "main" or "subtopic",
  "parent": "..." or null,
  "description": "one sentence description",
  "common_subtopics": ["...", "..."]
}}"""

            generation_params = {
                "temperature": 0.2,  # Low temperature for consistency
                "client_identifier": "backend-core-devs",
                "provider": "OPEN_AI",
                "max_tokens": 400
            }
            
            try:
                response = await llm_caller.call_and_get_text(
                    prompt=prompt,
                    model=LLM_MODEL,
                    generation_params=generation_params
                )
                
                # Parse JSON response (handle potential markdown wrapper)
                response = response.strip()
                if response.startswith("```"):
                    # Remove markdown code block
                    lines = response.split("\n")
                    response = "\n".join([l for l in lines if not l.startswith("```")])
                
                topic_info = json.loads(response)
                
                # Add metadata
                topic_info["variants"] = [t for t, _ in topic_variants]
                topic_info["total_occurrences"] = sum(count for _, count in topic_variants)
                topic_info["num_variants"] = len(topic_variants)
                
                return (cluster_id, topic_info)
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
                # Fallback: use most common variant
                most_common_topic = topic_variants[0][0]
                return (cluster_id, {
                    "canonical_name": most_common_topic,
                    "type": "main",
                    "parent": None,
                    "description": "Topic cluster",
                    "common_subtopics": [],
                    "variants": [t for t, _ in topic_variants],
                    "total_occurrences": sum(count for _, count in topic_variants),
                    "num_variants": len(topic_variants)
                })
    
    tasks = [process_cluster(cid, variants) for cid, variants in clusters.items()]
    results = []
    
    for i in range(0, len(tasks), 10):
        batch = tasks[i:i+10]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"Processed {min(i+10, len(tasks))}/{len(tasks)} clusters")
    
    canonical_topics = dict(results)
    
    return canonical_topics


def build_topic_taxonomy(canonical_topics: dict, embeddings: np.ndarray, unique_topics: list):
    """Build final taxonomy with IDs and embeddings."""
    
    # Sort topics by total occurrences
    sorted_topics = sorted(
        canonical_topics.items(),
        key=lambda x: x[1]['total_occurrences'],
        reverse=True
    )
    
    # Assign topic IDs (T001, T002, etc.)
    taxonomy = {}
    main_topics = []
    subtopics = []
    
    for idx, (cluster_id, topic_info) in enumerate(sorted_topics):
        topic_id = f"T{idx+1:03d}"
        
        # Compute average embedding for this topic cluster
        # Find indices of all variants in unique_topics
        variant_indices = [unique_topics.index(v) for v in topic_info['variants'] if v in unique_topics]
        avg_embedding = np.mean(embeddings[variant_indices], axis=0).tolist()
        
        taxonomy[topic_id] = {
            **topic_info,
            "id": topic_id,
            "embedding": avg_embedding
        }
        
        if topic_info['type'] == 'main':
            main_topics.append(topic_id)
        else:
            subtopics.append(topic_id)
    
    return taxonomy, main_topics, subtopics


def print_taxonomy(taxonomy: dict, main_topics: list, subtopics: list):
    """Pretty print the taxonomy."""
    print(f"\nCanonical topic taxonomy:")
    print(f"Main topics: {len(main_topics)}, Subtopics: {len(subtopics)}")
    
    print(f"\nTop 15 main topics:")
    for topic_id in main_topics[:15]:
        topic = taxonomy[topic_id]
        print(f"  {topic_id}: {topic['canonical_name']}")
        print(f"    Coverage: {topic['total_occurrences']} mentions, {topic['num_variants']} variants")
        print(f"    Description: {topic['description']}")


async def create_main_topic_hierarchy(taxonomy: dict, target_main_topics: int = 12):
    """
    Second-pass consolidation: Group 200+ topics into ~10-15 main topics.
    Uses data-driven clustering without hardcoded category suggestions.
    
    Args:
        taxonomy: The existing taxonomy with ~225 topics
        target_main_topics: Target number of main topic categories (default 12)
    
    Returns:
        Updated taxonomy with hierarchical structure
    """
    llm_caller = SimpleLLMCaller(max_retries=3, semaphore=10)
    
    print(f"\nCreating hierarchical consolidation: {target_main_topics} main topics from {len(taxonomy)} topics")
    
    topic_ids = list(taxonomy.keys())
    topic_names = [taxonomy[tid]['canonical_name'] for tid in topic_ids]
    embeddings = np.array([taxonomy[tid]['embedding'] for tid in topic_ids])
    
    clustering = AgglomerativeClustering(
        n_clusters=target_main_topics,
        metric='cosine',
        linkage='average'
    )
    main_cluster_labels = clustering.fit_predict(embeddings)
    
    main_clusters = {}
    for topic_id, cluster_label in zip(topic_ids, main_cluster_labels):
        if cluster_label not in main_clusters:
            main_clusters[cluster_label] = []
        main_clusters[cluster_label].append(topic_id)
    
    print(f"Grouped into {len(main_clusters)} main clusters")
    sem = asyncio.Semaphore(5)
    
    async def create_main_topic(cluster_id, subtopic_ids):
        async with sem:
            # Get all subtopics info sorted by occurrence
            subtopics_info = [(tid, taxonomy[tid]) for tid in subtopic_ids]
            subtopics_info.sort(key=lambda x: x[1]['total_occurrences'], reverse=True)
            
            # Include ALL topic names with their occurrence counts
            # This gives the LLM the full picture of what's in this cluster
            topic_list = [
                f"  - {info['canonical_name']} ({info['total_occurrences']} occurrences)"
                for tid, info in subtopics_info[:20]  # Top 20 for context
            ]
            
            # Also include some variants to show the diversity
            variant_examples = []
            for tid, info in subtopics_info[:5]:
                if info.get('variants'):
                    variant_examples.append(f"    [{info['canonical_name']} includes: {', '.join(info['variants'][:3])}]")
            
            prompt = f"""You are analyzing Statement of Work (SOW) document topics to create a taxonomy.

Below are {len(subtopics_info)} semantically similar topics that were AUTOMATICALLY CLUSTERED together:

{chr(10).join(topic_list)}

Example variants showing the diversity within these topics:
{chr(10).join(variant_examples[:3])}

Task: Based ONLY on these clustered topics, create:
1. A BROAD, HIGH-LEVEL main category name (2-3 words) that encompasses ALL these topics
2. A clear description of what this category represents
3. Identify the 3-5 key thematic areas within this cluster

Guidelines:
- The name should be general enough to cover all topics shown
- Look for the common thread that connects these topics
- Use professional SOW/contract language
- DO NOT invent categories - base your answer purely on what you see above

Return ONLY valid JSON (no markdown):
{{
  "main_topic_name": "...",
  "description": "one sentence describing this broad category",
  "key_areas": ["3-5 key areas this covers based on the topics listed"]
}}"""

            generation_params = {
                "temperature": 0.3,
                "client_identifier": "backend-core-devs",
                "provider": "OPEN_AI",
                "max_tokens": 300
            }
            
            try:
                response = await llm_caller.call_and_get_text(
                    prompt=prompt,
                    model=LLM_MODEL,
                    generation_params=generation_params
                )
                
                # Parse response
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join([l for l in lines if not l.startswith("```")])
                
                main_info = json.loads(response)
                return (cluster_id, main_info, subtopic_ids)
                
            except Exception as e:
                print(f"Error creating main topic {cluster_id}: {e}")
                top_topic = subtopics_info[0][1]
                return (cluster_id, {
                    "main_topic_name": top_topic['canonical_name'],
                    "description": top_topic['description'],
                    "key_areas": []
                }, subtopic_ids)
    
    tasks = [create_main_topic(cid, subs) for cid, subs in main_clusters.items()]
    results = await asyncio.gather(*tasks)
    
    new_taxonomy = {}
    main_topic_ids = []
    
    for idx, (cluster_id, main_info, subtopic_ids) in enumerate(sorted(results, key=lambda x: len(x[2]), reverse=True)):
        main_topic_id = f"M{idx+1:02d}"
        main_topic_ids.append(main_topic_id)
        
        total_occurrences = sum(taxonomy[tid]['total_occurrences'] for tid in subtopic_ids)
        
        new_taxonomy[main_topic_id] = {
            "canonical_name": main_info['main_topic_name'],
            "type": "main",
            "parent": None,
            "description": main_info['description'],
            "key_areas": main_info.get('key_areas', []),
            "subtopics": subtopic_ids,
            "total_occurrences": total_occurrences,
            "num_subtopics": len(subtopic_ids),
            "id": main_topic_id
        }
        
        for subtopic_id in subtopic_ids:
            new_taxonomy[subtopic_id] = {
                **taxonomy[subtopic_id],
                "type": "subtopic",
                "parent": main_topic_id
            }
        
        print(f"{main_topic_id}: {main_info['main_topic_name']} ({len(subtopic_ids)} subtopics, {total_occurrences} occurrences)")
    
    return new_taxonomy, main_topic_ids


def save_hierarchical_taxonomy(taxonomy: dict, main_topics: list, output_file: str):
    """Save the hierarchical taxonomy with visual formatting."""
    
    subtopics_by_parent = {}
    for topic_id, topic_info in taxonomy.items():
        if topic_info['type'] == 'subtopic':
            parent = topic_info['parent']
            if parent not in subtopics_by_parent:
                subtopics_by_parent[parent] = []
            subtopics_by_parent[parent].append(topic_id)
    
    output = {
        "main_topics": {},
        "metadata": {
            "num_main_topics": len(main_topics),
            "num_total_topics": len(taxonomy),
            "model": LLM_MODEL,
            "created_at": "2025-11-01"
        }
    }
    
    for main_id in main_topics:
        main_topic = taxonomy[main_id]
        subtopic_ids = subtopics_by_parent.get(main_id, [])
        
        subtopics_data = []
        for sub_id in subtopic_ids:
            sub_info = taxonomy[sub_id]
            subtopics_data.append({
                "id": sub_id,
                "name": sub_info['canonical_name'],
                "description": sub_info['description'],
                "occurrences": sub_info['total_occurrences'],
                "variants": sub_info['variants'][:5]
            })
        subtopics_data.sort(key=lambda x: x['occurrences'], reverse=True)
        
        output["main_topics"][main_id] = {
            "id": main_id,
            "name": main_topic['canonical_name'],
            "description": main_topic['description'],
            "key_areas": main_topic.get('key_areas', []),
            "total_occurrences": main_topic['total_occurrences'],
            "subtopics": subtopics_data
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved hierarchical taxonomy to: {output_file}")
    
    full_output_file = output_file.replace('.json', '_full.json')
    with open(full_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "taxonomy": taxonomy,
            "main_topics": main_topics,
            "metadata": output["metadata"]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved full taxonomy to: {full_output_file}")


def print_hierarchical_taxonomy(taxonomy: dict, main_topics: list):
    """Pretty print the hierarchical taxonomy."""
    subtopics_by_parent = {}
    for topic_id, topic_info in taxonomy.items():
        if topic_info['type'] == 'subtopic':
            parent = topic_info['parent']
            if parent not in subtopics_by_parent:
                subtopics_by_parent[parent] = []
            subtopics_by_parent[parent].append((topic_id, topic_info))
    
    print(f"\nHierarchical taxonomy: {len(main_topics)} main topics\n")
    
    for main_id in main_topics:
        main_topic = taxonomy[main_id]
        print(f"{main_id}: {main_topic['canonical_name']}")
        print(f"  {main_topic['description']}")
        print(f"  Coverage: {main_topic['total_occurrences']} occurrences, {main_topic['num_subtopics']} subtopics")
        
        if main_id in subtopics_by_parent:
            subtopics = subtopics_by_parent[main_id]
            subtopics.sort(key=lambda x: x[1]['total_occurrences'], reverse=True)
            
            for sub_id, sub_info in subtopics[:5]:
                print(f"    - {sub_info['canonical_name']} ({sub_info['total_occurrences']} occurrences)")
        print()


async def main(similarity_threshold: float = 0.75, create_hierarchy: bool = True, target_main_topics: int = 12):
    """
    Main function to consolidate LLM-extracted topics.
    
    Args:
        similarity_threshold: Similarity threshold for clustering (0-1, default 0.75)
        create_hierarchy: Whether to create hierarchical main topics (default True)
        target_main_topics: Number of main topic categories for hierarchy (default 12)
    """
    # Hardcoded paths
    input_file = "./extracted_topics/sow_topics.json"
    output_file = "./extracted_topics/canonical_topics.json"
    hierarchical_output = "./extracted_topics/hierarchical_topics.json"
    
    sow_topics, topic_freq = load_and_analyze_topics(input_file)
    
    clusters, embeddings, unique_topics = cluster_similar_topics(
        topic_freq, 
        similarity_threshold=similarity_threshold
    )
    
    canonical_topics = await generate_canonical_names(clusters)
    
    taxonomy, main_topics, subtopics = build_topic_taxonomy(
        canonical_topics, 
        embeddings, 
        unique_topics
    )
    
    print_taxonomy(taxonomy, main_topics, subtopics)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "taxonomy": taxonomy,
            "main_topics": main_topics,
            "subtopics": subtopics,
            "metadata": {
                "num_source_documents": len(sow_topics),
                "num_canonical_topics": len(taxonomy),
                "similarity_threshold": similarity_threshold,
                "model": LLM_MODEL
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved initial canonical taxonomy to: {output_file}")
    
    if create_hierarchy:
        hierarchical_taxonomy, hierarchical_main_topics = await create_main_topic_hierarchy(
            taxonomy, 
            target_main_topics=target_main_topics
        )
        
        print_hierarchical_taxonomy(hierarchical_taxonomy, hierarchical_main_topics)
        
        save_hierarchical_taxonomy(
            hierarchical_taxonomy, 
            hierarchical_main_topics, 
            hierarchical_output
        )
        
        variant_to_hierarchical = {}
        for topic_id, topic_info in hierarchical_taxonomy.items():
            if topic_info['type'] == 'subtopic':
                for variant in topic_info.get('variants', []):
                    variant_to_hierarchical[variant] = {
                        "subtopic_id": topic_id,
                        "subtopic_name": topic_info['canonical_name'],
                        "main_topic_id": topic_info['parent'],
                        "main_topic_name": hierarchical_taxonomy[topic_info['parent']]['canonical_name']
                    }
        
        hierarchical_mapping_file = hierarchical_output.replace('.json', '_mapping.json')
        with open(hierarchical_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(variant_to_hierarchical, f, indent=2, ensure_ascii=False)
        
        print(f"Saved hierarchical mapping to: {hierarchical_mapping_file}")
    
    variant_to_canonical = {}
    for topic_id, topic_info in taxonomy.items():
        for variant in topic_info.get('variants', []):
            variant_to_canonical[variant] = {
                "canonical_id": topic_id,
                "canonical_name": topic_info['canonical_name']
            }
    
    mapping_file = output_file.replace('.json', '_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(variant_to_canonical, f, indent=2, ensure_ascii=False)
    
    print(f"Saved variant-to-canonical mapping to: {mapping_file}")


if __name__ == "__main__":
    # You can adjust these parameters:
    # - similarity_threshold: 0.7-0.8 for more/fewer initial clusters
    # - target_main_topics: 10-15 for number of main categories
    asyncio.run(main(similarity_threshold=0.75, create_hierarchy=True, target_main_topics=12))