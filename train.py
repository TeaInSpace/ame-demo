import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
import os


if __name__ == "__main__":
    print("db secret: ", os.environ.get("DB_SECRET"))

    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model", registered_model_name="logreg")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
