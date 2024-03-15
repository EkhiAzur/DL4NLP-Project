from load_dataset import load_dataset
from load_model import load_model
from contrastive_trainer import ConstTrainer
from runArgs import runArgs
from transformers import TrainingArguments, Trainer, HfArgumentParser

def mymetric(p):
    predictions, labels = p

    if type(predictions) == tuple:
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main(runArgs, trainingArgs, *args, **kwargs):

    logging.info("Starting training!")

    # Setting seed
    seed = trainingArgs.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.info("Loadding model and tokenizer")

    # Load model and tokenizer
    model, tokenizer = load_model(runArgs, trainingArgs)

    # Load dataset

    logging.info("Loadding dataset")

    train_dataset, eval_dataset, test_dataset = load_dataset(runArgs, trainingArgs, tokenizer)

    # Training

    if runArgs.contrastive:
        trainer = ConstTrainer(
            temp = runArgs.contrastive_temp,
            lam = runArgs.contrastive_lam,
            model=model,
            args=trainingArgs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics = myetric,
        )
        
    else:

        trainer = Trainer(
            model=model,
            args=trainingArgs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics = myetric,
        )

    trainer.train()

    trainer.save_model()

    # Test the model on the splits

    dev_results = trainer.evaluate(eval_dataset)

    test_results = trainer.evaluate(test_dataset)
    
    train_results = trainer.evaluate(train_dataset)

    logging.info(f"Train results: {train_results}")

    logging.info(f"Dev results: {dev_results}")

    logging.info(f"Test results: {test_results}")


if __name__ == '__main__':

    parser = HfArgumentParser((runArgs, TrainingArguments))

    runArgs, trainingArgs = parser.parse_args_into_dataclasses()

    if runArgs.contrastive:

        runArgs.label2id = {"txiki": 0, "wiki": 1}
        runArgs.id2label = {0: "txiki", 1: "wiki"}

    else:
        
        runArgs.label2id = {"EZ_GAI": 0, "GAI": 1}
        runArgs.id2label = {0: "EZ_GAI", 1: "GAI"}

    main(runArgs, trainingArgs)
