import wandb
from typing import Union, List, Tuple, Callable
from torch.utils.data import Dataset

from .utils import get_instance_shape

def log_dataset_artifact(
    project: str,
    dataset_name: str,
    category: str,
    stage: int, 
    version: str,
    description: str,
    source: str,

    features_shape: Union[List, Tuple],
    labels_shape: Union[List, Tuple],

    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,

    info_function: Callable = None,
    *args, **kwargs,
) -> None:

    assert stage in [1, 2, 3, 4], f'\n\nStage must be from 1 to 4.'

    if version != 1:
        assert stage == 4, f'\n\nDataset versioning, (version > 0) is only allowed at stage 4.\n'

    with wandb.init(
        project = project,
        job_type = f'dataset-stage({stage})',
        name = f'dataset-stage{stage}' if stage != 4 else f'dataset-stage{stage}-v{version}',
        group = 'dataset',
        id = str(hash(f'{project}-{dataset_name}-{stage}-{version}'))
    ) as run:

        if stage > 1 and stage != 4:
            run.use_artifact(f'{dataset_name}-stage{stage - 1}:latest')
        elif version != 1:
            run.use_artifact(f'{dataset_name}-stage{stage}-v{version - 1}:latest')

        artifact = wandb.Artifact(
            name = f'{dataset_name}-stage{stage}' if stage != 4 else f'{dataset_name}-stage{stage}-v{version}',
            type = "dataset",
            description = description,
            metadata = {
                "source": source,
                "category": category,
                
                "training samples": len(train_dataset),
                "validation samples": len(val_dataset),
                "testing samples": len(test_dataset),
                
                "features shape": get_instance_shape(features_shape),
                "labels shape": get_instance_shape(labels_shape),
            },
        )

        if info_function:
            print('\n[==================== INFO ====================]\n')
            info_function(*args, **kwargs)
            print('\n[==============================================]\n')

        run.log_artifact(artifact)