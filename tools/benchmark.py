#!/usr/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import json
import os
import sys
import subprocess


class Benchmark():
    def __init__(self, model_name, input_len, output_len, num_run, num_decoder_layer, num_header, size_per_header, max_batch_size, vocab_size):
        self.model_name = model_name
        self.input_len = input_len
        self.output_len = output_len
        self.num_run = num_run
        self.num_decoder_layer = num_decoder_layer
        self.num_header = num_header
        self.size_per_header = size_per_header
        self.gpu_mem_footprint = []
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.server_log = f"{self.model_name}_inference_server.log"
        self.client_log = f"{self.model_name}_client.log"
        self.data_points = []

    def parse_log(self, batch_size):
        cmd = f"tail -n 1 {self.client_log} | grep -Eo '[+-]?[0-9]+([.][0-9]+)?'"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        avg_latency = float(out.strip())
        cmd = f"tail -2 {self.client_log} | head -n 1"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(out, err)
        latencies = ["{:.2f}".format(float(n)) for n in out.decode('utf-8').strip('\n').strip(']').strip('[').split(", ")]
       

        #TODO: get mem usage
        cmd = f"(cat {self.server_log} | grep 'before allocation' | sort | awk '{{print $8}}' )"
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(out, err)
        before_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]


        cmd = f"(cat {self.server_log} | grep 'after allocation' | sort | awk '{{print $8}}' )"
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        print(out, err)
        after_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]
        assert len(after_gpu_mem) == len(before_gpu_mem)
        print(before_gpu_mem)
        print(after_gpu_mem)
        gpu_mem_usage = ["{:.2f}".format(b - a) for b, a in zip(before_gpu_mem, after_gpu_mem)]
        data_point = [batch_size] + latencies + [avg_latency] + gpu_mem_usage 
        self.data_points.append(data_point)


    def call_once(self, batch_size):
        #cmd = f"bash {os.getenv('WORKSPACE')}/fastertransformer_backend/tools/benchmark_single_node.sh -b {batch_size} -m {self.model_name} -i {self.input_len} -o {self.output_len} -d {self.num_decoder_layer} -h_n {self.num_header} -s_h {self.size_per_header} -v {self.vocab_size} -n {self.num_run}"
        #print(cmd)
        #proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #out, err = proc.communicate()
        self.parse_log(batch_size)
        print(self.data_points)

    def start(self):
        os.getenv('WORKSPACE')
        bs = 1
        while (bs <= self.max_batch_size):
            self.call_once(bs)
            bs = bs * 2

    def to_csv(self):
        import csv

        with open(f"{self.model_name}_perf.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.data_points)  

if __name__ == '__main__':
    #b = Benchmark("8-gpu", 512, 32 , 10, 24, 32, 64, 64, 50304)
    #b = Benchmark("3-gpu", 512, 32 , 10, 3, 32, 64, 2, 50304)
    b = Benchmark("89B", 512, 32 , 10, 48, 96, 128, 64, 51200)

    b.start()
    #b.to_csv()