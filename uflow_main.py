from functools import partial

import numpy
from absl import app
from absl import flags

import numpy as np
import os
import torch
import uflow_augmentation
import uflow_data
import uflow_flags
import uflow_net
import uflow_plotting
from uflow_net import UFlow
import uflow_gpu_utils

FLAGS = flags.FLAGS

def create_uflow():
  """Build the uflow model."""

  build_selfsup_transformations = partial(
      uflow_augmentation.build_selfsup_transformations,
      crop_height=FLAGS.selfsup_crop_height,
      crop_width=FLAGS.selfsup_crop_width,
      max_shift_height=FLAGS.selfsup_max_shift,
      max_shift_width=FLAGS.selfsup_max_shift,
      resize=FLAGS.resize_selfsup)

  # Define learning rate schedules [none, cosine, linear, expoential].
  def learning_rate_fn():
    # step = tf.compat.v1.train.get_or_create_global_step()
    effective_step = torch.maximum(torch.tensor(uflow_flags.step) - torch.tensor(FLAGS.lr_decay_after_num_steps) + 1, torch.tensor(0))
    lr_step_ratio = effective_step.type(torch.float32) / float(FLAGS.lr_decay_steps)
    if FLAGS.lr_decay_type == 'none' or FLAGS.lr_decay_steps <= 0:
      return FLAGS.gpu_learning_rate
    elif FLAGS.lr_decay_type == 'cosine':
      x = np.pi * torch.minimum(lr_step_ratio, torch.tensor(1.0))
      return FLAGS.gpu_learning_rate * (torch.cos(x) + 1.0) / 2.0
    elif FLAGS.lr_decay_type == 'linear':
      return FLAGS.gpu_learning_rate * torch.maximum(1.0 - lr_step_ratio, torch.tensor(0.0))
    elif FLAGS.lr_decay_type == 'exponential':
      return FLAGS.gpu_learning_rate * 0.5**lr_step_ratio
    else:
      raise ValueError('Unknown lr_decay_type', FLAGS.lr_decay_type)

  occ_weights = {
      'fb_abs': FLAGS.occ_weights_fb_abs,
      'forward_collision': FLAGS.occ_weights_forward_collision,
      'backward_zero': FLAGS.occ_weights_backward_zero,
  }
  # Switch off loss-terms that have weights < 1e-2.
  occ_weights = {k: v for (k, v) in occ_weights.items() if v > 1e-2}

  occ_thresholds = {
      'fb_abs': FLAGS.occ_thresholds_fb_abs,
      'forward_collision': FLAGS.occ_thresholds_forward_collision,
      'backward_zero': FLAGS.occ_thresholds_backward_zero,
  }
  occ_clip_max = {
      'fb_abs': FLAGS.occ_clip_max_fb_abs,
      'forward_collision': FLAGS.occ_clip_max_forward_collision,
  }

  uflow = UFlow(
      checkpoint_dir=FLAGS.checkpoint_dir,
      summary_dir=FLAGS.tensorboard_logdir,
      optimizer=FLAGS.optimizer,
      learning_rate=learning_rate_fn(),
      only_forward=FLAGS.only_forward,
      level1_num_layers=FLAGS.level1_num_layers,
      level1_num_filters=FLAGS.level1_num_filters,
      level1_num_1x1=FLAGS.level1_num_1x1,
      dropout_rate=FLAGS.dropout_rate,
      build_selfsup_transformations= build_selfsup_transformations,
      fb_sigma_teacher=FLAGS.fb_sigma_teacher,
      fb_sigma_student=FLAGS.fb_sigma_student,
      train_with_supervision=FLAGS.use_supervision,
      train_with_gt_occlusions=FLAGS.use_gt_occlusions,
      smoothness_edge_weighting=FLAGS.smoothness_edge_weighting,
      teacher_image_version=FLAGS.teacher_image_version,
      stop_gradient_mask=FLAGS.stop_gradient_mask,
      selfsup_mask=FLAGS.selfsup_mask,
      normalize_before_cost_volume=FLAGS.normalize_before_cost_volume,
      original_layer_sizes=FLAGS.original_layer_sizes,
      shared_flow_decoder=FLAGS.shared_flow_decoder,
      channel_multiplier=FLAGS.channel_multiplier,
      num_levels=FLAGS.num_levels,
      use_cost_volume=FLAGS.use_cost_volume,
      use_feature_warp=FLAGS.use_feature_warp,
      accumulate_flow=FLAGS.accumulate_flow,
      occlusion_estimation=FLAGS.occlusion_estimation,
      occ_weights=occ_weights,
      occ_thresholds=occ_thresholds,
      occ_clip_max=occ_clip_max,
      smoothness_at_level=FLAGS.smoothness_at_level,
  )

  return uflow

def check_model_frozen(feature_model, flow_model, prev_flow_output=None):
  """Check that a frozen model isn't somehow changing over time."""
  state = np.random.RandomState(40)
  input1 = state.randn(FLAGS.batch_size, 3, FLAGS.height, FLAGS.width).astype(np.float32)
  input2 = state.randn(FLAGS.batch_size, 3, FLAGS.height, FLAGS.width).astype(np.float32)
  feature_output1 = feature_model(input1, split_features_by_sample=False)
  feature_output2 = feature_model(input2, split_features_by_sample=False)
  flow_output = flow_model(feature_output1, feature_output2, training=False)
  if prev_flow_output is None:
    return flow_output
  for f1, f2 in zip(prev_flow_output, flow_output):
    assert np.max(f1.numpy() - f2.numpy()) < .01

def create_frozen_teacher_models(uflow):
  """Create a frozen copy of the current uflow model."""
  uflow_copy = create_uflow()
  teacher_feature_model = uflow_copy.feature_model
  teacher_flow_model = uflow_copy.flow_model
  # need to create weights in teacher models by calling them
  bogus_input1 = np.random.randn(FLAGS.batch_size, 3, FLAGS.height, FLAGS.width).astype(np.float32)
  bogus_input2 = np.random.randn(FLAGS.batch_size, 3, FLAGS.height, FLAGS.width).astype(np.float32)
  existing_model_output = uflow.feature_model( bogus_input1, split_features_by_sample=False)
  _ = teacher_feature_model(bogus_input1, split_features_by_sample=False)
  teacher_flow_model.load_state_dict(uflow.feature_model.state_dict())
  teacher_output1 = teacher_feature_model(bogus_input1, split_features_by_sample=False)
  teacher_output2 = teacher_feature_model(bogus_input2, split_features_by_sample=False)

  # check that both feature models have the same output
  assert np.max(existing_model_output[-1].numpy() - teacher_output1[-1].numpy()) < .01
  existing_model_flow = uflow.flow_model(teacher_output1, teacher_output2, training=False)
  _ = teacher_flow_model(teacher_output1, teacher_output2, training=False)
  teacher_flow_model.load_state_dict(uflow.flow_model.state_dict())
  teacher_flow = teacher_flow_model( teacher_output1, teacher_output2, training=False)
  # check that both flow models have the same output
  assert np.max(existing_model_flow[-1].numpy() - teacher_flow[-1].numpy()) < .01
  # Freeze the teacher models.
  for para in teacher_feature_model.parameters():
      para.requires_grad = False
  for para in teacher_flow_model.parameters():
      para.requires_grad = False

  return teacher_feature_model, teacher_flow_model

def main(unused_argv):
  uflow_gpu_utils.setup_gpu()

  # gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  # # Make directories if they do not exist yet.
  if FLAGS.checkpoint_dir and not os.path.exists(FLAGS.checkpoint_dir):
    print('Making new checkpoints directory', FLAGS.checkpoint_dir)
    os.makedirs(FLAGS.checkpoint_dir)
  if FLAGS.plot_dir and not os.path.exists(FLAGS.plot_dir):
    print('Making new plot directory', FLAGS.plot_dir)
    os.makedirs(FLAGS.plot_dir)

  uflow = create_uflow().to(uflow_gpu_utils.device)

  if not FLAGS.from_scratch:
    # First restore from init_checkpoint_dir, which is only restored from but
    # not saved to, and then restore from checkpoint_dir if there is already
    # a model there (e.g. if the run was stopped and restarted).
    if FLAGS.init_checkpoint_dir:
      print('Initializing model from checkpoints {}.'.format(
          FLAGS.init_checkpoint_dir))
      uflow.update_checkpoint_dir(FLAGS.init_checkpoint_dir)
      uflow.restore(
          reset_optimizer=FLAGS.reset_optimizer,
          reset_global_step=FLAGS.reset_global_step)
      uflow.update_checkpoint_dir(FLAGS.checkpoint_dir)

    if FLAGS.checkpoint_dir:
      print('Restoring model from checkpoints {}.'.format(os.path.join(FLAGS.checkpoint_dir,_) for _ in os.listdir(FLAGS.checkpoint_dir) if _.endswith(".pth")))
      if len(os.listdir(FLAGS.checkpoint_dir)) == 0:
          print('Checkpoint directory is empty, nothing is restored')
      else:
          checkpoints_list = [os.path.join(FLAGS.checkpoint_dir, _) for _ in os.listdir(FLAGS.checkpoint_dir) if _.endswith(".pth")]
          checkpoints = torch.load(checkpoints_list[-1])
          uflow.load_state_dict(checkpoints["model_state_dict"])
          uflow._optimizer.load_state_dict(checkpoints["optimizer"])
          uflow.restore(steps= checkpoints["epoch"])
          uflow.train()

  else:
    print('Starting from scratch.')

  print('Making eval datasets and eval functions.')
  if FLAGS.eval_on:
    evaluate, _ = uflow_data.make_eval_function(
        FLAGS.eval_on,
        FLAGS.height,
        FLAGS.width,
        progress_bar=True,
        plot_dir=FLAGS.plot_dir,
        num_plots=50)

  if FLAGS.train_on:
    # Build training iterator.
    print('Making training iterator.')
    train_it = uflow_data.make_train_iterator(
        FLAGS.train_on,
        FLAGS.height,
        FLAGS.width,
        FLAGS.shuffle_buffer_size,
        FLAGS.batch_size,
        FLAGS.seq_len,
        FLAGS.entire_sequence_for_kitti,
        crop_instead_of_resize=FLAGS.crop_instead_of_resize,
        apply_augmentation=True,
        include_ground_truth=FLAGS.use_supervision,
        resize_gt_flow=FLAGS.resize_gt_flow_supervision,
        include_occlusions=FLAGS.use_gt_occlusions,

    )
    if FLAGS.use_supervision:
      # Since this is the only loss in this setting, and the Adam optimizer
      # is scale invariant, the actual weight here does not matter for now.
      weights = {'supervision': 1.}
    else:
      # Note that self-supervision loss is added during training.
      weights = {
          'photo': FLAGS.weight_photo,
          'ssim': FLAGS.weight_ssim,
          'census': FLAGS.weight_census,
          'smooth1': FLAGS.weight_smooth1,
          'smooth2': FLAGS.weight_smooth2,
          'edge_constant': FLAGS.smoothness_edge_constant,
      }

      # Switch off loss-terms that have weights < 1e-7.
      weights = {
          k: v for (k, v) in weights.items() if v > 1e-7 or k == 'edge_constant'
      }

    def weight_selfsup_fn():

      step = uflow_flags.step % FLAGS.selfsup_step_cycle
      # Start self-supervision only after a certain number of steps.
      # Linearly increase self-supervision weight for a number of steps.
      ramp_up_factor = torch.clamp(torch.tensor(
          float(step - (FLAGS.selfsup_after_num_steps - 1)) /
          float(max(FLAGS.selfsup_ramp_up_steps, 1))), 0., 1.)
      return FLAGS.weight_selfsup * ramp_up_factor

    distance_metrics = {
        'photo': FLAGS.distance_photo,
        'census': FLAGS.distance_census,
    }

    print('Starting training loop.')
    log = dict()
    epoch = 0

    teacher_feature_model = None
    teacher_flow_model = None
    test_frozen_flow = None

    while True:
      current_step = numpy.array(uflow_flags.step)

      # Set which occlusion estimation methods could be active at this point.
      # (They will only be used if occlusion_estimation is set accordingly.)
      occ_active = {
          'uflow':
              FLAGS.occlusion_estimation == 'uflow',
          'brox':
              current_step > FLAGS.occ_after_num_steps_brox,
          'wang':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wang4':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wangthres':
              current_step > FLAGS.occ_after_num_steps_wang,
          'wang4thres':
              current_step > FLAGS.occ_after_num_steps_wang,
          'fb_abs':
              current_step > FLAGS.occ_after_num_steps_fb_abs,
          'forward_collision':
              current_step > FLAGS.occ_after_num_steps_forward_collision,
          'backward_zero':
              current_step > FLAGS.occ_after_num_steps_backward_zero,
      }

      current_weights = {k: v for k, v in weights.items()}

      # Prepare self-supervision if it will be used in the next epoch.
      if FLAGS.weight_selfsup > 1e-7 and (
          current_step % FLAGS.selfsup_step_cycle
      ) + FLAGS.epoch_length > FLAGS.selfsup_after_num_steps:

        # Add selfsup weight with a ramp-up schedule. This will cause a
        # recompilation of the training graph defined in uflow.train(...).
        current_weights['selfsup'] = weight_selfsup_fn


        # Freeze model for teacher distillation.
        if teacher_feature_model is None and FLAGS.frozen_teacher:
          # Create a copy of the existing models and freeze them as a teacher.
          # Tell uflow about the new, frozen teacher model.
          teacher_feature_model, teacher_flow_model = create_frozen_teacher_models(
              uflow)
          uflow.set_teacher_models(
              teacher_feature_model=teacher_feature_model,
              teacher_flow_model=teacher_flow_model)
          test_frozen_flow = check_model_frozen(
              teacher_feature_model, teacher_flow_model, prev_flow_output=None)

          # Check that the model actually is frozen.
          if FLAGS.frozen_teacher and test_frozen_flow is not None:
            check_model_frozen(
                teacher_feature_model,
                teacher_flow_model,
                prev_flow_output=test_frozen_flow)


      # Train for an epoch and save the results.

      uflow.train()
      log_update = uflow.trainer(train_it, num_steps=FLAGS.epoch_length, weights=current_weights, progress_bar=True,
                                  plot_dir=FLAGS.plot_dir if FLAGS.plot_debug_info else None,
                                  distance_metrics=distance_metrics, occ_active=occ_active)
      # print(uflow.state_dict())

      for key in log_update:
        if key in log:
          log[key].append(log_update[key])
        else:
          log[key] = [log_update[key]]

      if FLAGS.checkpoint_dir and not FLAGS.no_checkpointing:


        """Saves a model checkpoints."""
        torch.save({
            "epoch": uflow_flags.step // FLAGS.epoch_length,
            "optimizer": uflow._optimizer.state_dict(),
            "model_state_dict": uflow.state_dict(),
        }, FLAGS.checkpoint_dir + "/" + FLAGS.checkpoint_dir + " .pth")


      # Print losses from last epoch.
      uflow_plotting.print_log(log, epoch)

      if epoch % 10 == 0:
          if FLAGS.eval_on and FLAGS.evaluate_during_train:
            # Evaluate
            eval_results = evaluate(uflow)
            uflow_plotting.print_eval(eval_results)

      if current_step >= FLAGS.num_train_steps:
        break

      epoch += 1

  else:
    print('Specify flag train_on to enable training to <format>:<path>;... .')
    print('Just doing evaluation now.')
    eval_results = evaluate(uflow)
    if eval_results:
        uflow_plotting.print_eval(eval_results)
    print('Evaluation complete.')

if __name__ == '__main__':
  app.run(main)

# --train_on="kitti:datasets/home/manish/winshare/datasets/data_scene_flow_multiview"