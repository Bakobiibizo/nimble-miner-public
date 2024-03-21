from execute import execute

task_args = {
    "model_name": "bert-base-uncased",
    "num_labels": 5000,
    "num_rows": 5000,
    "dataset_name": "yelp_review_full"
    }

def main(task_args):
    return execute(task_args)

if __name__ == "__main__":
    main(task_args)