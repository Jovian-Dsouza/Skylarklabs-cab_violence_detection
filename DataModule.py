import os
from glob import glob
import pytorch_lightning
import pytorchvideo.data
from pytorchvideo.data import labeled_video_dataset
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip
)

def get_count(paths, category):
    count = 0 
    for path in paths:
        if category in path:
            count += 1
    return count

def get_category(path):
    return path.split(os.path.sep)[-2]

class WeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source) -> None:
        n_samples = len(data_source)
        paths = [data_source[i][0] for i in range(n_samples)]
        category_list = list(set([get_category(path) for path in paths]))
        
        category_weights = {category: (1/get_count(paths, category)) for category in category_list}
        sample_weights = [category_weights[get_category(path)] for path in paths]
        
        self.sampler =  torch.utils.data.WeightedRandomSampler(
                            sample_weights,
                            num_samples = n_samples,
                            replacement = True
                            )

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return self.sampler.__len__()

class VideoDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_path, clip_duration, batch_size, num_workers=None, pin_memory=False):
        '''
        data_path - Dataset path containing "train", "test" and "val"
        clip_duration - (in s) Duration of sampled clip for each video
        batch_size 
        num_workers -  Number of parallel processes fetching data
        pin_memory 
        '''
        super().__init__()

        self._DATA_PATH = data_path
        self._CLIP_DURATION = clip_duration  
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers  
        self._PIN_MEMORY = pin_memory

        if num_workers is None:
            self._NUM_WORKERS = os.cpu_count()-1

        # Transform parameters
        # mean = (0.45, 0.45, 0.45)
        # std = (0.225, 0.225, 0.225)
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        crop = 128
        pad = 10

        self.transform_dict = {
            'train' : Compose(
                        [
                        ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [   
                                Resize((crop+pad, crop+pad)),
                                RandomCrop((crop, crop)),
                                Lambda(lambda x: x / 255.0),
                                Normalize(mean, std),
                                RandomHorizontalFlip(p=0.5),
                                RandomVerticalFlip(p=0.5)
                            ]
                            ),
                        ),
                        ]
                    ),
            'val' : Compose(
                        [
                        ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [   
                                Resize((crop, crop)),
                                Lambda(lambda x: x / 255.0),
                                Normalize(mean, std),
                            ]
                            ),
                        ),
                        ]
                    ),
            'test' : Compose(
                        [
                        ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [   
                                Resize((crop, crop)),
                                Lambda(lambda x: x / 255.0),
                                Normalize(mean, std)         
                            ]
                            ),
                        ),
                        ]
                    )
        }

    def train_dataloader(self):
        """
        Create the train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        dataset =  labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, 'train'),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            video_sampler=WeightedRandomSampler,
            transform=self.transform_dict['train'],
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory = self._PIN_MEMORY
        )

    def val_dataloader(self):
        """
        Create the train partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        dataset =  labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, 'val'),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=self.transform_dict['val'],
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory = self._PIN_MEMORY
        )
    
    def test_dataloader(self):
        """
        Create the train partition from the list of video labels
        in {self._DATA_PATH}/test
        """
        dataset =  labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, 'test'),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=self.transform_dict['test'],
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory = self._PIN_MEMORY
        )

    def __repr__(self):
        train_paths = glob(os.path.join(self._DATA_PATH, 'train', '*', '*'))
        val_paths = glob(os.path.join(self._DATA_PATH, 'val', '*', '*'))
        test_paths = glob(os.path.join(self._DATA_PATH, 'test', '*', '*'))

        category_index = os.listdir(os.path.join(self._DATA_PATH, 'train'))

        rep_str = '.........................................\n'
        
        for category in category_index:
            training_frac  = get_count(train_paths, category)
            validation_frac  = get_count(val_paths, category)
            testing_frac  = get_count(test_paths, category)
            total_videos = training_frac + validation_frac + testing_frac

            rep_str += (f'\n{total_videos} videos for "{category}"\n')
            rep_str += (f'\tTraining: {training_frac}\n')
            rep_str += (f'\tValidation: {validation_frac}\n')
            rep_str += (f'\tTesting: {testing_frac}\n')
            rep_str += ('.........................................\n')

        rep_str += f'\nTraining videos: {len(train_paths)}\n'
        rep_str += f'Validation videos: {len(val_paths)}\n'
        rep_str += f'Testing videos: {len(test_paths)}\n'
        rep_str += '.........................................\n'

        return rep_str

if __name__ == '__main__':
    from pprint import pprint 

    data_module = VideoDataModule(
        data_path = '../../Dataset/CAR_VIOLENCE_DATASET_final', 
        clip_duration = 3.2,
        batch_size = 8,
        num_workers = 8,
        pin_memory = False)

    print(data_module)

   