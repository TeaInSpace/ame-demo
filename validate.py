import sys
import mlflow
from mlflow.entities import ViewType


results = mlflow.search_registered_models(filter_string="name='logreg'")
run_id = results[0].latest_versions[-1].run_id
run = mlflow.get_run(run_id)
print(run.data.metrics['score'])

if run.data.metrics['score'] < 0.6:
    sys.exit(1)
