import os
import sys

import mlflow


def start(experiment, name, params=None):
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:7461"))
        exp = mlflow.set_experiment(experiment)
        found = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{name}'", output_format="list", max_results=1)
        if found:
            return mlflow.start_run(run_id=found[0].info.run_id)
        run = mlflow.start_run(run_name=name)
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        return run
    except Exception as e:
        print(f"mlflow start failed: {e}", file=sys.stderr, flush=True)


def log(rec, step):
    if "eval_ema" in rec:
        metrics = {f"eval/{k}": v for k, v in rec["eval_ema"].items()}
    else:
        metrics = {k: v for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "step"}
    if metrics and mlflow.active_run():
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"mlflow log failed: {e}", file=sys.stderr, flush=True)
