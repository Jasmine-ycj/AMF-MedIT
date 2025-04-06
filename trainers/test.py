from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint


def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.eval_datatype == 'imaging':
    test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, hparams.img_size, target=hparams.target, train=False, live_loading=hparams.live_loading)
    print(test_dataset.transform_val.__repr__())
  elif hparams.eval_datatype == 'tabular':
    test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = test_dataset.get_input_size()
  elif hparams.eval_datatype == 'imaging_and_tabular':
    test_dataset = ImagingAndTabularDataset(hparams.data_test_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate,  hparams.data_test_eval_tabular,
      hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target)
    hparams.input_size = test_dataset.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular or imaging_and_tabular')
  
  drop = ((len(test_dataset)%hparams.batch_size)==1)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  hparams.dataset_length = len(test_loader)

  model = Evaluator(hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
  trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)
  if hparams.eval_metric == 'auc':
    return model.auc_test.compute()
  elif hparams.eval_metric == 'acc':
    return model.acc_test.compute(), model.acc_test_topk.compute()