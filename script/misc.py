from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_prompt(prompt):
    '''
    Classify the prompt using zero-shot classification
    '''
    # Classify the prompt
    labels = ["furniture", "vehicles", "characters", "person", 'other']
    classification = classifier(prompt, candidate_labels=labels)
    
    ## Return the top label
    return classification['labels'][0]


if __name__ == "__main__":
    # Example text
    texts = ["A modern sofa", "A new chair", "Table Tennis racket", "A fancy new car"]
    labels = ["furniture", "vehicles", "characters", "person", 'other']

    for text in texts:
        print(classify_prompt(text))
