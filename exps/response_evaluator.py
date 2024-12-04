import pandas as pd
from evaluate import load
from bert_score import score
import textstat
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle


def evaluate_summary_quality(generated_summary, reference_summary):
    rouge = load("rouge")
    rouge_score = rouge.compute(predictions=[generated_summary], references=[reference_summary])

    P, R, F1 = score([generated_summary], [reference_summary], lang="en")
    bert_score = F1.mean().item()

    return {
        "ROUGE": rouge_score,
        "BERTScore F1": bert_score
    }


def evaluate_document_complexity(text):
    """
    Calculate a simple complexity score based on lexical and structural features.

    Args:
        text (str): The document text.

    Returns:
        int: A score representing the document's complexity.
    """
    words = text.split()
    sentences = text.split('.')

    # Lexical features
    avg_word_length = (sum(len(word) for word in words) + 1) / (len(words) + 1)
    type_token_ratio = (len(set(words)) + 1) / (len(words) + 1)

    # Sentence features
    avg_sentence_length = (len(words) + 1) / (len(sentences) + 1)

    # Flesch-Kincaid Readability Score (lower score indicates higher complexity)
    fk_score = textstat.flesch_reading_ease(text)

    # Combine features into a weighted score (tune weights as needed)
    return avg_word_length, type_token_ratio, avg_sentence_length, fk_score


def create_dataset():
    print('---------X-Sum--------')
    df = pd.read_parquet('dataset/x-sum.parquet')
    random_sample = df.sample(n=8759, random_state=42)
    # Initialize the new DataFrame
    new_data = {
        "avg_word_length": [],
        "average_TTR": [],
        "average_sentence_length": [],
        "avg_FK_Score": [],
        "complexity": []
    }
    # Compute metrics for each row in the sample
    for text in random_sample['text']:
        avg_word_length, avg_ttr, avg_sentence_length, avg_fk_score = evaluate_document_complexity(text)
        new_data["avg_word_length"].append(avg_word_length)
        new_data["average_TTR"].append(avg_ttr)
        new_data["average_sentence_length"].append(avg_sentence_length)
        new_data["avg_FK_Score"].append(avg_fk_score)
        new_data["complexity"].append("easy")
    print('---------Arxiv--------')
    df = pd.read_parquet('dataset/arxiv.parquet')
    random_sample = df.sample(n=8759, random_state=42)
    # Compute metrics for each row in the sample
    for text in random_sample['article']:
        avg_word_length, avg_ttr, avg_sentence_length, avg_fk_score = evaluate_document_complexity(text)
        new_data["avg_word_length"].append(avg_word_length)
        new_data["average_TTR"].append(avg_ttr)
        new_data["average_sentence_length"].append(avg_sentence_length)
        new_data["avg_FK_Score"].append(avg_fk_score)
        new_data["complexity"].append("medium")

    print('---------Gov-Report------')
    df = pd.read_parquet('dataset/gov-report.parquet')
    random_sample = df.sample(n=8759, random_state=42)
    # Compute metrics for each row in the sample
    for text in random_sample['report']:
        avg_word_length, avg_ttr, avg_sentence_length, avg_fk_score = evaluate_document_complexity(text)
        new_data["avg_word_length"].append(avg_word_length)
        new_data["average_TTR"].append(avg_ttr)
        new_data["average_sentence_length"].append(avg_sentence_length)
        new_data["avg_FK_Score"].append(avg_fk_score)
        new_data["complexity"].append("difficult")
    new_df = pd.DataFrame(new_data)
    new_df.to_csv('complexity_stats.csv')

def train_model():
    df = pd.read_csv('dataset/complexity_stats.csv')
    df['complexity'] = df['complexity'].map({'easy': 0, 'medium': 1, 'difficult': 2})

    # Define features (X) and target (y)
    X = df[["avg_word_length", "TTR", "average_sentence_length", "FK_Score"]]
    y = df["complexity"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save the trained llm as a pickle file
    with open("../local/server/src/main/models/classifier/logistic_regression_model.pkl", "wb") as model_file:
        pickle.dump(clf, model_file)

    print("Model saved as 'logistic_regression_model.pkl'.")


if __name__ == "__main__":
    # Load the ROUGE metric
    #
    # rouge = load('rouge')
    #
    # # Example: evaluate a generated summary against a reference summary
    # generated_summary = "The llm generated summary."
    # reference_summary = "The reference summary from the dataset."
    #
    # results = evaluate_summary_quality(generated_summary, reference_summary)
    # create_dataset()
    train_model()
