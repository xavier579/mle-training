import mlflow

from my_package.ingest_data import Ingest
from my_package.train import Train

# Create nested runs
EXP_NAME = "Housing_Price_Prediction"
remote_server_uri = "http://localhost:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
print(mlflow.tracking.get_tracking_uri())
mlflow.set_experiment(EXP_NAME)

with mlflow.start_run(
    run_name="PARENT_RUN",
    # experiment_id=experiment_id,
    tags={"version": "v1"},
    description="parent",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN_INGEST",
        # experiment_id=experiment_id,
        description="Ingest Data",
        nested=True,
    ) as child_run:
        global opt
        mlflow.log_param("child", "yes")
        ing = Ingest()
        opt = ing.parse_opt()
        ing.main(opt)
    with mlflow.start_run(
        run_name="CHILD_RUN_TRAIN",
        # experiment_id=experiment_id,
        description="Train Model",
        nested=True,
    ) as child_run:
        mlflow.log_param("child", "yes")
        # mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
        tr = Train()
        tr.main()

    # with mlflow.start_run(
    #     run_name="CHILD_RUN_SCORE",
    #     # experiment_id=experiment_id,
    #     description="Evaluate",
    #     nested=True,
    # ) as child_run:
    #     mlflow.log_param("child", "yes")
    #     scr = Score()
    #     scr.main()

print("parent run:")
print("run_id: {}".format(parent_run.info.run_id))
print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
print("version tag value: {}".format(parent_run.data.tags.get("version")))
print("--")

# Search all child runs with a parent id
query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
results = mlflow.search_runs(filter_string=query)
print("child runs:")
print(results[["run_id", "params.child", "tags.mlflow.runName"]])
