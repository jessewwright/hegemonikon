import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def process_framing_sentences(sentences: list[str]) -> dict[str, float]:
    """
    Processes framing sentences to calculate valence scores using a sentiment analysis model.

    Args:
        sentences: A list of framing sentences (strings).

    Returns:
        A dictionary mapping each original sentence to its final valence_score.
    """
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    if not sentences:
        return {}

    tanh_logits = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Assuming we take the logit for the "positive" class or a specific one if multi-class.
            # For simplicity, let's assume the model's primary logit output is what we need.
            # If the model outputs multiple logits (e.g., for negative, neutral, positive),
            # we might need to select or combine them.
            # For "cardiffnlp/twitter-roberta-base-sentiment", the output logits are typically [negative, neutral, positive].
            # Let's consider the difference between positive and negative sentiment as a single score,
            # or simply use the raw logits if the model is trained for regression-like sentiment.
            # Given the problem doesn't specify, let's use the first logit output directly and apply tanh.
            # A more robust approach might be to use (positive_logit - negative_logit).
            # For now, let's use the mean of the logits as a single value before tanh.
            # Or, let's assume the "score" is represented by the first logit for simplicity,
            # acknowledging this might need refinement based on model specifics.
            # Upon checking the model card for cardiffnlp/twitter-roberta-base-sentiment,
            # it outputs 3 scores: 0 -> negative, 1 -> neutral, 2 -> positive.
            # For "cardiffnlp/twitter-roberta-base-sentiment", the output logits are typically [negative, neutral, positive].
            # We derive a single semantic logit representing overall valence: positive_logit - negative_logit.
            # This single value is "the logit" for the sentence in terms of its valence.
            semantic_logit = logits[0, 2] - logits[0, 0]  # Positive logit - Negative logit
            
            # Step 2c: Apply a tanh function to this semantic logit.
            # This becomes the sentence's "tanh-transformed logit".
            sentence_tanh_logit = torch.tanh(semantic_logit)
            tanh_logits.append(sentence_tanh_logit.item())

    if not tanh_logits: # Should not happen if sentences is not empty, but as a safeguard
        return {sentence: 0.0 for sentence in sentences}

    mean_tanh_logit = np.mean(tanh_logits)

    preliminary_valence_scores = {}
    for i, sentence in enumerate(sentences):
        preliminary_valence_scores[sentence] = tanh_logits[i] - mean_tanh_logit

    max_abs_preliminary_score = 0.0
    if preliminary_valence_scores: # Check if dictionary is not empty
        max_abs_preliminary_score = np.max(np.abs(list(preliminary_valence_scores.values())))

    final_valence_scores = {}
    if max_abs_preliminary_score == 0: # Avoid division by zero if all scores are the same
        for sentence in sentences:
            final_valence_scores[sentence] = 0.0
    else:
        for sentence, prelim_score in preliminary_valence_scores.items():
            final_valence_scores[sentence] = prelim_score / max_abs_preliminary_score
            
    return final_valence_scores

if __name__ == "__main__":
    example_sentences = [
        "This is a wonderful experience!",
        "I am not happy with the outcome.",
        "It's an okay day, nothing special.",
        "The new policy is absolutely fantastic and will benefit everyone.",
        "Many are concerned about the recent changes."
    ]
    
    # Ensure transformers and torch are installed
    # You might need to run: pip install torch transformers
    
    print("Processing example sentences...")
    results = process_framing_sentences(example_sentences)
    
    print("\nResults:")
    for sentence, score in results.items():
        print(f"Sentence: \"{sentence}\" \nValence Score: {score:.4f}\n")

    example_sentences_2 = [
        "Climate change is a hoax.",
        "Urgent action on climate change is needed to save the planet."
    ]
    print("Processing second set of example sentences...")
    results_2 = process_framing_sentences(example_sentences_2)
    print("\nResults for second set:")
    for sentence, score in results_2.items():
        print(f"Sentence: \"{sentence}\" \nValence Score: {score:.4f}\n")

    example_sentences_3 = ["All is well.", "All is well."]
    print("Processing third set of example sentences (identical)...")
    results_3 = process_framing_sentences(example_sentences_3)
    print("\nResults for third set:")
    for sentence, score in results_3.items():
        print(f"Sentence: \"{sentence}\" \nValence Score: {score:.4f}\n")

    example_sentences_4 = []
    print("Processing fourth set of example sentences (empty)...")
    results_4 = process_framing_sentences(example_sentences_4)
    print("\nResults for fourth set:")
    for sentence, score in results_4.items():
        print(f"Sentence: \"{sentence}\" \nValence Score: {score:.4f}\n")
