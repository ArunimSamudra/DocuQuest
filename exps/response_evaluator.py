from evaluate import load
from bert_score import score

# Load the ROUGE metric

rouge = load('rouge')

# Example: evaluate a generated summary against a reference summary
generated_summary = "The model generated summary."
reference_summary = "The reference summary from the dataset."

results = rouge.compute(predictions=[generated_summary], references=[reference_summary])

# Display ROUGE scores
print(results)

# Calculate BERTScore
P, R, F1 = score([generated_summary], [reference_summary], lang="en")

# Display the F1 score, which combines precision and recall
print(f"BERTScore F1: {F1.mean().item()}")


def evaluate_summary_quality(generated_summary, reference_summary):
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=[generated_summary], references=[reference_summary])

    P, R, F1 = score([generated_summary], [reference_summary], lang="en")
    bert_score = F1.mean().item()

    return {
        "ROUGE": rouge_score,
        "BERTScore F1": bert_score
    }
