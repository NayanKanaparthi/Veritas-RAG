"""Synthetic corpus generator for benchmarking."""

import json
import random
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from veritas_rag.ingestion.normalizer import normalize_text


def _generate_vocabulary(vocab_size: int, seed: Optional[int] = None) -> List[str]:
    """Generate a vocabulary pool of words."""
    if seed is not None:
        random.seed(seed)
    words = []
    for i in range(vocab_size):
        # Generate words of varying lengths
        length = random.randint(3, 10)
        word = "".join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    return words


def _generate_entities(num_entities: int, seed: Optional[int] = None) -> List[str]:
    """Generate entity names (proper nouns)."""
    if seed is not None:
        random.seed(seed)
    entities = []
    prefixes = ["John", "Mary", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    suffixes = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore"]
    
    for i in range(num_entities):
        if i < len(prefixes) * len(suffixes):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[(i // len(prefixes)) % len(suffixes)]
            entities.append(f"{prefix} {suffix}")
        else:
            # Generate random entity names
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            entities.append(f"{prefix} {suffix}")
    
    return entities


def _generate_exact_phrases(num_phrases: int, vocab: List[str], seed: Optional[int] = None) -> List[str]:
    """Generate exact phrases for injection."""
    if seed is not None:
        random.seed(seed)
    phrases = []
    for i in range(num_phrases):
        # Generate 2-4 word phrases
        phrase_length = random.randint(2, 4)
        phrase_words = random.sample(vocab, min(phrase_length, len(vocab)))
        phrases.append(" ".join(phrase_words))
    return phrases


# Fixed synonym map for deterministic paraphrasing
_SYNONYM_MAP = {
    "project": "initiative",
    "latency": "delay",
    "budget": "allocation",
    "system": "platform",
    "performance": "efficiency",
    "optimization": "improvement",
    "algorithm": "method",
    "data": "information",
    "model": "framework",
    "training": "learning",
}

# Fixed stopwords list
_STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}


def _generate_paraphrase(phrase: str, seed: int) -> str:
    """
    Generate deterministic paraphrase using fixed transforms.
    
    Args:
        phrase: Original phrase
        seed: Random seed for reproducibility
    
    Returns:
        Paraphrased phrase
    """
    random.seed(seed)
    words = phrase.split()
    
    # Apply transforms in fixed order
    result_words = []
    
    # 1. Synonym replacement (deterministic)
    for word in words:
        word_lower = word.lower()
        if word_lower in _SYNONYM_MAP:
            result_words.append(_SYNONYM_MAP[word_lower])
        else:
            result_words.append(word)
    
    # 2. Remove stopwords (deterministic)
    result_words = [w for w in result_words if w.lower() not in _STOPWORDS]
    
    # 3. Word order swap (deterministic: swap adjacent pairs)
    if len(result_words) >= 2:
        # Swap pairs deterministically based on seed
        swapped = []
        i = 0
        while i < len(result_words):
            if i + 1 < len(result_words) and random.random() < 0.5:  # Deterministic with seed
                swapped.append(result_words[i + 1])
                swapped.append(result_words[i])
                i += 2
            else:
                swapped.append(result_words[i])
                i += 1
        result_words = swapped
    
    # 4. Question template variants (deterministic)
    if random.random() < 0.3:  # Deterministic with seed
        return f"What is {result_words[0] if result_words else phrase}?"
    
    return " ".join(result_words) if result_words else phrase


def generate_synthetic_corpus(
    output_dir: Path,
    target_chunks: int = 10000,
    num_docs: Optional[int] = None,
    vocab_size: int = 1000,
    entity_injection_rate: float = 0.1,
    exact_phrase_rate: float = 0.05,
    distractor_rate: float = 0.0,
    chunk_size: int = 512,
    seed: Optional[int] = None,
) -> Tuple[Path, Dict]:
    """
    Generate synthetic corpus for benchmarking with exact offset tracking.

    Args:
        output_dir: Directory to write corpus files
        target_chunks: Target number of chunks (approximate)
        num_docs: Number of documents (auto-calculated if None)
        vocab_size: Size of vocabulary pool
        entity_injection_rate: Rate of entity injection (proper nouns)
        exact_phrase_rate: Rate of exact phrase injection
        distractor_rate: Rate of distractor content (for harder queries)
        chunk_size: Approximate words per chunk
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corpus_path, metadata_dict) where metadata contains injections with exact offsets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)

    # Calculate number of documents if not provided
    if num_docs is None:
        # Estimate: each doc produces roughly (doc_words / chunk_size) chunks
        # For simplicity, assume each doc has ~2-3 chunks worth of words
        words_per_doc = chunk_size * 2.5
        num_docs = max(1, int(target_chunks * chunk_size / words_per_doc))

    # Generate vocabulary and special content
    vocab = _generate_vocabulary(vocab_size, seed)
    num_entities = int(vocab_size * entity_injection_rate)
    entities = _generate_entities(num_entities, seed)
    num_phrases = int(vocab_size * exact_phrase_rate)
    exact_phrases = _generate_exact_phrases(num_phrases, vocab, seed)

    # Track injections with exact offsets
    injections = []
    words_generated = 0
    target_words = target_chunks * chunk_size

    for doc_idx in range(num_docs):
        doc_words = []
        doc_length = random.randint(chunk_size, chunk_size * 3)
        source_path = f"doc_{doc_idx:05d}.txt"

        # Track injections: (category, original_quote, unique_quote, word_index_in_doc)
        # For exact-phrase: unique_quote has unique tag, original_quote is the base phrase
        # For other categories: unique_quote == original_quote
        injection_tracking = []  # List of (category, original_quote, unique_quote, word_index)
        injection_idx = 0  # Track injection index per document for unique tagging

        # Generate base text from vocabulary
        for word_idx in range(doc_length):
            if random.random() < entity_injection_rate:
                # Inject entity
                entity = random.choice(entities)
                doc_words.append(entity)
                injection_tracking.append(("entity-heavy", entity, entity, word_idx))
            elif random.random() < exact_phrase_rate:
                # Inject exact phrase with unique tag
                phrase = random.choice(exact_phrases)
                unique_tag = f"DOC{doc_idx:05d}_INJ{injection_idx:04d}"
                unique_phrase = f"{unique_tag} {phrase}"
                doc_words.append(unique_phrase)
                injection_tracking.append(("exact-phrase", phrase, unique_phrase, word_idx))
                injection_idx += 1
            else:
                # Use regular vocabulary
                word = random.choice(vocab)
                doc_words.append(word)

        # Build raw text with sentence structure
        raw_text = ""
        for i, word in enumerate(doc_words):
            if i > 0 and i % 15 == 0:
                raw_text += ". "
            raw_text += word + " "
        raw_text = raw_text.strip()

        # Normalize text (same as ingestion pipeline)
        normalized_text = normalize_text(raw_text)

        # Map injections to normalized text offsets
        # For each injection, find where it appears in normalized_text
        for category, original_quote, unique_quote, word_idx in injection_tracking:
            # Use unique_quote for offset finding (the one actually in the document)
            quote_normalized = normalize_text(unique_quote)
            
            # Find the quote in normalized_text
            # Try to find it by searching from the beginning, but we need to account for
            # the fact that word positions might have shifted due to normalization
            # Strategy: find all occurrences and pick the one that makes sense given word_idx
            
            # Simple approach: find first occurrence (works if quote is unique enough)
            offset_start = normalized_text.find(quote_normalized)
            if offset_start == -1:
                # Fallback: try to find by individual words
                quote_words = quote_normalized.split()
                if quote_words:
                    # Find first word
                    first_word = quote_words[0]
                    offset_start = normalized_text.find(first_word)
                    if offset_start != -1:
                        # Find end by searching for last word after first word
                        search_start = offset_start
                        for word in quote_words[1:]:
                            word_pos = normalized_text.find(word, search_start)
                            if word_pos != -1:
                                search_start = word_pos
                            else:
                                break
                        # Estimate end position
                        if search_start > offset_start:
                            offset_end = search_start + len(quote_words[-1])
                        else:
                            offset_end = offset_start + len(quote_normalized)
                    else:
                        continue  # Skip if we can't find it
                else:
                    continue
            else:
                offset_end = offset_start + len(quote_normalized)

            # Record injection with exact offsets
            # Store both original_quote (for paraphrase queries) and unique_quote (for exact queries)
            injection_record = {
                "source_path": source_path,
                "category": category,
                "original_quote": original_quote,
                "unique_quote": unique_quote,
                "relevant_quote": unique_quote,  # For backward compatibility, use unique_quote
                "offset_start": offset_start,
                "offset_end": offset_end,
                "doc_id": None,  # Will be filled after build
            }
            injections.append(injection_record)

        # Write document
        doc_file = output_dir / source_path
        doc_file.write_text(raw_text)

        words_generated += len(doc_words)
        if words_generated >= target_words:
            break

    # Write corpus metadata
    corpus_meta = {
        "generator_config": {
            "vocab_size": vocab_size,
            "entity_injection_rate": entity_injection_rate,
            "exact_phrase_rate": exact_phrase_rate,
            "distractor_rate": distractor_rate,
            "seed": seed,
            "target_chunks": target_chunks,
            "chunk_size": chunk_size,
        },
        "injections": injections,
    }

    meta_path = output_dir / "corpus_meta.json"
    with open(meta_path, "w") as f:
        json.dump(corpus_meta, f, indent=2)

    return output_dir, corpus_meta


def generate_synthetic_eval_set(
    corpus_meta_path: Path, 
    output_path: Path, 
    seed: Optional[int] = None,
    max_queries: Optional[int] = 1000,
) -> Path:
    """
    Generate synthetic eval set from corpus metadata.
    
    Args:
        corpus_meta_path: Path to corpus_meta.json
        output_path: Path to write eval_set_synth.jsonl
        seed: Random seed for paraphrase generation
        max_queries: Maximum number of queries to generate (default: 1000 for faster benchmarking)
    
    Returns:
        Path to generated eval set file
    """
    # Load corpus metadata
    with open(corpus_meta_path, "r") as f:
        corpus_meta = json.load(f)
    
    generator_config = corpus_meta["generator_config"]
    injections = corpus_meta["injections"]
    
    # Use seed from generator config if not provided
    if seed is None:
        seed = generator_config.get("seed", 42)
    random.seed(seed)
    
    queries = []
    query_id_counter = 1
    
    # Group injections by source_path for multi-hop detection
    injections_by_doc = {}
    for injection in injections:
        source_path = injection["source_path"]
        if source_path not in injections_by_doc:
            injections_by_doc[source_path] = []
        injections_by_doc[source_path].append(injection)
    
    # Generate queries from injections
    used_for_multi_hop = set()
    
    for injection in injections:
        category = injection["category"]
        quote = injection["relevant_quote"]
        source_path = injection["source_path"]
        
        # Skip if already used in multi-hop
        injection_key = (source_path, injection["offset_start"], injection["offset_end"])
        if injection_key in used_for_multi_hop:
            continue
        
        if category == "exact-phrase":
            # Exact phrase query: use the unique phrase (with tag) as query text
            # This ensures BM25 can reliably find the correct document
            unique_quote = injection.get("unique_quote", injection.get("relevant_quote"))
            query_text = unique_quote
            relevant_items = [{
                "source_path": source_path,
                "relevant_quote": unique_quote,  # Store unique quote for matching
                "offset_start": injection["offset_start"],
                "offset_end": injection["offset_end"],
                "relevance_score": 1,
            }]
            
        elif category == "entity-heavy":
            # Entity query: use the entity as query
            query_text = quote
            relevant_items = [{
                "source_path": source_path,
                "relevant_quote": quote,
                "offset_start": injection["offset_start"],
                "offset_end": injection["offset_end"],
                "relevance_score": 1,
            }]
            
        else:
            # Default: treat as exact-phrase
            query_text = quote
            relevant_items = [{
                "source_path": source_path,
                "relevant_quote": quote,
                "offset_start": injection["offset_start"],
                "offset_end": injection["offset_end"],
                "relevance_score": 1,
            }]
        
        queries.append({
            "query_id": f"synth_q{query_id_counter:03d}",
            "query_text": query_text,
            "category": category,
            "relevant_items": relevant_items,
        })
        query_id_counter += 1
        
        # Try to create paraphrase query
        if category in ["exact-phrase", "entity-heavy"]:
            # Use original_quote (without unique tag) for paraphrase generation
            # This tests semantic matching, not exact phrase matching
            original_quote = injection.get("original_quote", injection.get("relevant_quote"))
            paraphrase = _generate_paraphrase(original_quote, seed + query_id_counter)
            # For paraphrase queries, use unique_quote in relevant_items (the one in the document)
            unique_quote = injection.get("unique_quote", injection.get("relevant_quote"))
            queries.append({
                "query_id": f"synth_q{query_id_counter:03d}",
                "query_text": paraphrase,
                "category": "paraphrase",
                "relevant_items": [{
                    "source_path": source_path,
                    "relevant_quote": unique_quote,  # Use unique quote for matching
                    "offset_start": injection["offset_start"],
                    "offset_end": injection["offset_end"],
                    "relevance_score": 1,
                }],
            })
            query_id_counter += 1
    
    # Generate multi-hop queries (doc A has entity X, doc B has entity Y)
    # Find pairs of entities in different documents
    entity_injections = [inj for inj in injections if inj["category"] == "entity-heavy"]
    if len(entity_injections) >= 2:
        # Try to create a few multi-hop queries
        for _ in range(min(5, len(entity_injections) // 2)):
            if len(entity_injections) < 2:
                break
            
            # Pick two entities from different documents
            entity1 = random.choice(entity_injections)
            entity_injections.remove(entity1)
            
            # Find entity from different document
            entity2 = None
            for e in entity_injections:
                if e["source_path"] != entity1["source_path"]:
                    entity2 = e
                    entity_injections.remove(e)
                    break
            
            if entity2:
                # Create multi-hop query
                query_text = f"{entity1['relevant_quote']} and {entity2['relevant_quote']}"
                relevant_items = [
                    {
                        "source_path": entity1["source_path"],
                        "relevant_quote": entity1["relevant_quote"],
                        "offset_start": entity1["offset_start"],
                        "offset_end": entity1["offset_end"],
                        "relevance_score": 1,
                    },
                    {
                        "source_path": entity2["source_path"],
                        "relevant_quote": entity2["relevant_quote"],
                        "offset_start": entity2["offset_start"],
                        "offset_end": entity2["offset_end"],
                        "relevance_score": 1,
                    },
                ]
                
                queries.append({
                    "query_id": f"synth_q{query_id_counter:03d}",
                    "query_text": query_text,
                    "category": "multi-hop",
                    "relevant_items": relevant_items,
                })
                query_id_counter += 1
                
                used_for_multi_hop.add((entity1["source_path"], entity1["offset_start"], entity1["offset_end"]))
                used_for_multi_hop.add((entity2["source_path"], entity2["offset_start"], entity2["offset_end"]))
    
    # Limit queries if max_queries specified
    if max_queries and len(queries) > max_queries:
        # Sample proportionally from each category
        queries_by_category = defaultdict(list)
        for q in queries:
            category = q.get("category", "unknown")
            queries_by_category[category].append(q)
        
        sampled = []
        for category, cat_queries in queries_by_category.items():
            cat_max = max(1, int(max_queries * len(cat_queries) / len(queries)))
            sampled.extend(random.sample(cat_queries, min(cat_max, len(cat_queries))))
        
        # If we still have too many, randomly sample to reach exact max
        if len(sampled) > max_queries:
            sampled = random.sample(sampled, max_queries)
        
        queries = sampled
    
    # Write eval set
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")
    
    return output_path
