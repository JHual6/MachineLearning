from kedro.pipeline import Pipeline
from .pipelines.data_processing import pipeline as data_processing_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_processing_pipeline.create_pipeline(),
    }
