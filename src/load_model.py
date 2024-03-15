from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(runArgs, trainingArgs):

    tokenizer = AutoTokenizer.from_pretrained(
        runArgs.model_name,
        use_fast=True,
        add_prefix_space=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
            runArgs.model_name,
            num_labels=2,
            id2label = runArgs.id2label,
            label2id = runArgs.label2id,
    )

    return model, tokenizer