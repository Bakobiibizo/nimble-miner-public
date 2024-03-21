import datasets

# Define a simple dataset
data = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict({
        "text": ["This is a great movie.", "I didn't like the book.", "The food was delicious."],
        "label": [1, 0, 1]
    }),
    "test": datasets.Dataset.from_dict({
        "text": ["The weather is nice today.", "I love this song!"],
        "label": [1, 1]
    })
})