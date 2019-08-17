# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Executes a bisect."""
from __future__ import print_function

import argparse
import datetime
import json
import logging
import os
import random
import sys

import benchmark
import perfzero.perfzero_config as perfzero_config
import perfzero.utils as utils


class Bisect(object):
  """Execute bisect."""

  def __init__(self, starting_hash, ending_hash, tf_src_path, metric,
               low, high):
    self.starting_hash = starting_hash
    self.ending_hash = ending_hash
    self.tf_src_path = tf_src_path
    self.metric = metric
    if high:
      self.high = high
    else:
      self.high = float('inf')
    if low is None:
      self.low = float('-inf')
    else:
      self.low = low
    self.culprit_progress = None
    self.bazel_config_done = False
    # Used to get the perfzero output directory
    config_ = perfzero_config.PerfZeroConfig(mode='flags', flags=FLAGS)
    benchmark_runner2 = benchmark.BenchmarkRunner(config_)
    culprit_find_id = 'culprit_find-{}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
    self.culprit_output = os.path.join(benchmark_runner2.root_output_dir,
                                       culprit_find_id)
    utils.make_dir_if_not_exist(self.culprit_output)
    self.progress_file = os.path.join(self.culprit_output,
                                      'tracker.json')

  def checkout_tensorflow(self, git_hash):
    repo = {}
    repo['url'] = 'https://github.com/tensorflow/tensorflow.git'
    repo['local_path'] = self.tf_src_path
    repo['dir_name'] = 'tensorflow'
    repo['branch'] = git_hash
    utils.checkout_git_repos([repo], True)

  def build_and_install_tf(self, git_hash):
    # Clean environment and run configure configure with preset envvars.

    self.checkout_tensorflow(git_hash)
    if not self.bazel_config_done:
      os.environ['TF_NEED_CUDA'] = '1'
      os.environ['PYTHON_BIN_PATH'] = '/usr/bin/python3'
      config_cmd = ('cd {} && yes "" | ./configure').format(self.tf_src_path)
      self.bazel_config_done = True
      utils.run_commands([config_cmd])

    # Build Tensorflow and create wheel
    rm_whl_cmd = 'rm -rf /tmp/tensorflow_pkg'
    build_cmd = ('cd {} && bazel build --config=opt --config=v2 '
                 '//tensorflow/tools/pip_package:build_pip_package').format(
                     self.tf_src_path)
    build_pip_cmd = ('cd {} && ./bazel-bin/tensorflow/tools/pip_package/'
                     'build_pip_package --nightly_flag '
                     '/tmp/tensorflow_pkg').format(self.tf_src_path)
    utils.run_commands([rm_whl_cmd, build_cmd, build_pip_cmd])

    # Install TensorFlow
    install_tf_cmd = ('pip install --upgrade --force-reinstall '
                      '/tmp/tensorflow_pkg/tf*')
    utils.run_commands([install_tf_cmd])

    # Fixes estimator issue that often breaks running nightly TF.
    remove_tfe_cmd = 'pip uninstall -y tf-estimator-nightly'
    install_tfe_cmd = ('pip install --upgrade --force-reinstall '
                       'tensorflow-estimator-2.0-preview==1.14.0.dev2019073000')
    utils.run_commands([remove_tfe_cmd, install_tfe_cmd])

  def bisect_tensorflow(self):
    git_hashes = self.collect_hashes()

    # Stores initial status as not tested.
    for git_hash in git_hashes:
      self.update_progress(git_hash, 'not tested')

    # Updates assumptions to avoid confusion
    self.update_progress(git_hashes[0], 'Assume FAILED')
    self.update_progress(git_hashes[-1], 'Assume PASS')

    # Most recent known passing git_hash.
    bin_search_high = len(git_hashes) - 1
    # Most recent known failing git hash.
    bin_search_low = 0
    # Current hash being evaluated.
    bin_search_run = bin_search_high // 2
    print('Total hashes:{}'.format(bin_search_high))

    while True:
      test_hash = git_hashes[bin_search_run]
      print('Test Git Hash:[{}] {}'.format(bin_search_run, test_hash))
      self.build_and_install_tf(test_hash)

      test_result = self.run_and_check(test_hash)
      pass_fail = test_result['result']
      if pass_fail:
        print('Pass:[{}] {}'.format(bin_search_run, test_hash))
        self.update_progress(test_hash, 'PASS', result=test_result)
        bin_search_high = bin_search_run
        bin_search_run = bin_search_low + (
            (bin_search_run - bin_search_low) // 2)
      else:
        print('Fail:[{}] {}'.format(bin_search_run, test_hash))
        self.update_progress(test_hash, 'FAIL', result=test_result)
        bin_search_low = bin_search_run
        bin_search_run = bin_search_high - (
            (bin_search_high - bin_search_run) // 2)
      if bin_search_run == bin_search_low or bin_search_run == bin_search_high:

        print('Full report:/n{}'.format(json.dumps(self.culprit_progress,
                                                   ensure_ascii=False,
                                                   indent=2)))
        print('Culprit is [{}] {}'.format(bin_search_low,
                                          git_hashes[bin_search_low]))
        break

  def run_test(self):
    return bool(random.getrandbits(1))

  def run_and_check(self, git_hash):
    """Runs the test and returns a dictionary with result info.

    Args:
      git_hash: The TensorFlow githash under test.

    Returns:
      dict {
             result: boolean
             metric: str
             metric_value: double
             result_path: str
           }

    """
    result_path = self.run_perfzero(git_hash)
    result_file = os.path.join(result_path, 'result.json')
    result = {}
    result['metric'] = self.metric
    result['result_path'] = result_path
    if os.path.exists(result_file):
      with open(result_file, 'r') as f:
        try:
          results = json.load(f)
        except IOError as e:
          print('Error opening progress tracker:{}'.format(e))

    metric_value = None
    for metric in results['benchmark_result']['metrics']:
      if metric['name'] == self.metric:
        metric_value = metric['value']
        break
    result['metric_value'] = metric_value
    if metric_value:
      if metric_value <= self.high and metric_value >= self.low:
        result['result'] = True
      else:
        result['result'] = False
    else:
      print('Metric not found:{}'.format(self.metric))
      result['result'] = None
    return result

  def run_perfzero(self, git_hash):
    config_ = perfzero_config.PerfZeroConfig(mode='flags', flags=FLAGS)
    config_.execution_id = '{}-{}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'),
        git_hash)
    benchmark_runner2 = benchmark.BenchmarkRunner(config_)
    benchmark_runner2.run_benchmark()

    return os.path.join(benchmark_runner2.root_output_dir,
                        config_.execution_id)

  def collect_hashes(self):
    cmd = 'git -C {} log --pretty=oneline {}..{}'.format(self.tf_src_path,
                                                         self.starting_hash,
                                                         self.ending_hash)
    code, stdout = utils.run_command(cmd)
    git_hashes = []
    if code:
      print('Git range of hashes failed:{}'.format(stdout))
    else:
      print(stdout)
      lines = stdout.splitlines()
      for line in lines:
        parts = line.split(' ')
        git_hashes.append(parts[0])
    return git_hashes

  def update_progress(self, git_hash, status, result=None):
    self.load_stored_progress()
    test_status = {}
    test_status['status'] = status
    if result:
      test_status['metric'] = result['metric']
      test_status['metric_value'] = result['metric_value']
      test_status['log_path'] = result['result_path']
    if not self.culprit_progress:
      self.culprit_progress = {}
      self.culprit_progress['test_hashes'] = {}
    git_hash_statuses = self.culprit_progress['test_hashes']
    git_hash_statuses[git_hash] = test_status
    with open(self.progress_file, 'w', encoding='utf-8') as f:
      json.dump(self.culprit_progress, f, ensure_ascii=False, indent=2)

  def load_stored_progress(self):
    if os.path.exists(self.progress_file):
      with open(self.progress_file, 'r') as f:
        try:
          self.culprit_progress = json.load(f)
        except IOError as e:
          print('Error opening progress tracker:{}'.format(e))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  perfzero_config.add_benchmark_parser_arguments(parser)
  parser.add_argument(
      '--starting_hash',
      default=None,
      type=str,
      help='Git hash at which the test was passing.')
  parser.add_argument(
      '--ending_hash',
      default=None,
      type=str,
      help='Git hash at point where test was failing.')
  parser.add_argument(
      '--tensorflow_git_path',
      default='/tensorflow_src',
      type=str,
      help='Git hash at point where test was failing.')
  parser.add_argument(
      '--metric',
      default='exp_per_second',
      type=str,
      help='Metric to look for for success.')
  parser.add_argument(
      '--high',
      default=None,
      type=float,
      help='Highest acceptable value')
  parser.add_argument(
      '--low',
      default=None,
      type=float,
      help='Lowest acceptable value.')

  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    logging.error('Arguments %s are not recognized', unparsed)
    sys.exit(1)
  logging.basicConfig(
      format='%(asctime)s %(levelname)-8s %(message)s',
      level=logging.DEBUG,
      datefmt='%Y-%m-%d %H:%M:%S',
      filename='log.txt')
  benchmark_runner = Bisect(starting_hash=FLAGS.starting_hash,
                            ending_hash=FLAGS.ending_hash,
                            tf_src_path=FLAGS.tensorflow_git_path,
                            metric=FLAGS.metric,
                            low=FLAGS.low,
                            high=FLAGS.high)
  benchmark_runner.bisect_tensorflow()
