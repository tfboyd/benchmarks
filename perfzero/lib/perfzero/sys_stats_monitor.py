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
"""Gathers system stats."""
from __future__ import print_function

import argparse
import logging
import os
import time

import psutil


class StatEntry(object):

  def __init__(self, timestamp, value):
    self.timestamp = timestamp
    self.value = value


class SystemStatsMonitor(object):

  def __init__(self):
    self.memory_log = []

  def get_system_io_read(self):
    before = psutil.disk_io_counters()
    time.sleep(1)
    after = psutil.disk_io_counters()
    disks_read_per_sec = after.read_bytes - before.read_bytes
    return time.time(), disks_read_per_sec

  def get_process_memory(self, pid=None):
    # return the memory usage in percentage like top.
    if pid:
      process = psutil.Process(pid)
    else:
      process = psutil.Process(os.getpid())

    mem = process.memory_percent()
    return time.time(), mem


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_file',
      type=str,
      default=None,
      help='path to log stats.')

  parser.add_argument(
      '--log_interval',
      type=int,
      default=None,
      help='Interval in seconds to log stats.')

  parser.add_argument(
      '--process_id',
      type=int,
      default=None,
      help='Process to monitor, if not provided monitor self.')

  FLAGS, unparsed = parser.parse_known_args()

  logging.basicConfig(filename=FLAGS.log_file, level=logging.DEBUG)
  monitor = SystemStatsMonitor()
  while 2 > 1:
    _timestamp, _mem = monitor.get_process_memory(pid=FLAGS.process_id)
    logging.info('Process Memory:%s,%s', _timestamp, _mem)
    time.sleep(FLAGS.log_interval)

