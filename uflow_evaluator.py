from absl import app
from absl import flags
from absl import logging
import torch
import os

import uflow_data
import uflow_flags
import uflow_gpu_utils
import uflow_main
import uflow_plotting


FLAGS = flags.FLAGS



def evaluate():
  """Eval happens on GPU or CPU, and evals each checkpoints as it appears."""


  uflow = uflow_main.create_uflow().to(uflow_gpu_utils.device)
  evaluate_fn, _ = uflow_data.make_eval_function(
      FLAGS.eval_on,
      FLAGS.height,
      FLAGS.width,
      progress_bar=True,
      plot_dir=FLAGS.plot_dir,
      num_plots=200)

  checkpoints_list = [os.path.join(FLAGS.checkpoint_dir, _) for _ in os.listdir(FLAGS.checkpoint_dir) if
                      _.endswith(".pth")]
  latest_checkpoint = torch.load(checkpoints_list[-1])
  uflow.load_state_dict(latest_checkpoint["model_state_dict"])
  uflow._optimizer.load_state_dict(latest_checkpoint["optimizer"])
  uflow.restore(steps=latest_checkpoint["epoch"])
  uflow.eval()
  eval_results = evaluate_fn(uflow)
  uflow_plotting.print_eval(eval_results)



def main(unused_argv):


  # Make directories if they do not exist yet.
  if FLAGS.checkpoint_dir and not os.path.exists(FLAGS.checkpoint_dir):
      print("Checkpoint directory does not exists")
      return

  if FLAGS.plot_dir and not os.path.exists(FLAGS.plot_dir):
    logging.info('Making new plot directory %s', FLAGS.plot_dir)
    os.makedirs(FLAGS.plot_dir)

  if FLAGS.eval_on:
    evaluate()
  else:
    raise ValueError('evaluation needs --eval_on <dataset>.')


if __name__ == '__main__':
  app.run(main)