import wandb
import numpy as np
import tensorflow as tf
import argparse

TABLE_COLUMNS = ["image", "label", "prediction"]
TABLE_COLUMNS.extend([f"score_{i}" for i in range(10)])

def build_pred_table(dataset, logits):
    data = [
        [wandb.Image(dataset["x"][idx]), dataset["y"][idx], np.argmax(logits[idx]), *logits[idx]]
        for idx in range(len(logits))
    ]
    table = wandb.Table(data=data, columns=TABLE_COLUMNS)
    return table


def main():
    settings = wandb.Settings()
    settings.update({"enable_job_creation": True})
    default_config = {
        "model": "wandb-artifact://bcanfieldsherman/mnist-launch/mnist-model:latest",
        "dataset": "wandb-artifact://bcanfieldsherman/mnist-launch/mnist-data:latest"
    }
    with wandb.init(job_type="evaluate", config=default_config, settings=settings) as run:
        model = tf.keras.models.load_model(run.config.model.get_path("model").download())
        test_data = np.load(run.config.dataset.get_path("test").download())
        logits = model.predict(test_data["x"])
        preds = np.argmax(logits, axis=1)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(test_data["y"], logits).numpy()
        accuracy = tf.keras.metrics.Accuracy()(test_data["y"], preds).numpy()
        run.summary["loss"] = loss
        run.summary["accuracy"] = accuracy
        run.log({"predictions": build_pred_table(test_data, logits)})



if __name__ == "__main__":
    main()
