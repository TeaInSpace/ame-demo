import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os


if __name__ == "__main__":
    print("db secret: ", os.environ.get("DB_SECRET"))

    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    signature = infer_signature(train_x, predictions)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print("type: ", tracking_url_type_store)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model", registered_model_name="logreg", signature=signature)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
