#!/usr/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import time
from threading import Thread

class GPUUtilTracker():
    def __init__(self):
        self.max_gpu_mem_usage = []
        self.stop = False
    
    def get_results(self):
        return self.max_gpu_mem_usage

    def terminate(self):
        self.stop = True

    def run(self):
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        print(cmd)
        while(True):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            print(out, err)
            gpu_mem_usage = [float(n) for n in out.decode('utf-8').strip().split('\n')]
            print(gpu_mem_usage)
            if len(self.max_gpu_mem_usage) == 0:
                self.max_gpu_mem_usage = gpu_mem_usage
            else:
                for i in range(len(self.max_gpu_mem_usage)):
                    self.max_gpu_mem_usage[i] = gpu_mem_usage[i] if gpu_mem_usage[i] > self.max_gpu_mem_usage[i] else self.max_gpu_mem_usage[i]
            if self.stop:
                break
            time.sleep(5)


class Benchmark():
    def __init__(self, model_name, input_len, output_len, num_run, num_decoder_layer, num_header, size_per_header, max_batch_size, vocab_size, tensor_para_size=8):
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
        self.tensor_para_size = tensor_para_size
        self.server_log = f"{self.model_name}_inference_server.log"
        self.client_log = f"{self.model_name}_client.log"
        self.data_points = []

    def cal_num_params(self):
        hidden_size = self.num_header * self.size_per_header
        return 12 * self.num_decoder_layer * hidden_size * hidden_size * \
            (1 + 13 / (12 * hidden_size) + (self.vocab_size + 2048) / (12 * self.num_decoder_layer * hidden_size))

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
       
        # cmd = f"(cat {self.server_log} | grep 'after allocation' | sort | awk '{{print $8}}' )"
        # print(cmd)
        # proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # out, err = proc.communicate()
        # print(out, err)
        # free_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]


        # cmd = f"(cat {self.server_log} | grep 'after allocation' | sort | awk '{{print $11}}' )"
        # print(cmd)
        # proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # out, err = proc.communicate()
        # print(out, err)
        # total_gpu_mem = [float(n) for n in out.decode('utf-8').strip().split('\n')]
        # assert len(free_gpu_mem) == len(total_gpu_mem)
        # print(free_gpu_mem)
        # print(total_gpu_mem)
        # gpu_mem_usage = ["{:.2f}".format(b - a) for b, a in zip(total_gpu_mem, free_gpu_mem)]
        return [batch_size] + latencies + [avg_latency]

    def call_once(self, batch_size):
        g_tracker = GPUUtilTracker()
        t = Thread(target=g_tracker.run)
        t.start()
        devices = "CUDA_VISIBLE_DEVICES=" + ",".join([ str(i) for i in range(self.tensor_para_size) ])
        cmd = f"{devices} bash {os.getenv('WORKSPACE')}/fastertransformer_backend/tools/benchmark_single_node.sh -b {batch_size} -m {self.model_name} -i {self.input_len} -o {self.output_len} -d {self.num_decoder_layer} -h_n {self.num_header} -s_h {self.size_per_header} -v {self.vocab_size} -n {self.num_run} -t_p {self.tensor_para_size}"
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        bs_latency = self.parse_log(batch_size)
        g_tracker.terminate()
        t.join()
        self.data_points.append(bs_latency + g_tracker.get_results())

    def start(self):
        print("Estimated num param: ", "{:,}".format(int(self.cal_num_params())))
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
    
    # b = Benchmark("125M", 512, 32, 10, num_decoder_layer=12, num_header=12, size_per_header=64, max_batch_size=64, vocab_size=51200)
    # b.start()
    # b.to_csv()

    tensor_para_size = 1
    while (tensor_para_size <= 8):
        b = Benchmark(f"350M_TP_{tensor_para_size}", 512, 32, 10, num_decoder_layer=24, num_header=16, size_per_header=64, max_batch_size=64, vocab_size=51200, tensor_para_size=tensor_para_size)
        b.start()
        b.to_csv()
        tensor_para_size = tensor_para_size * 2

    # b = Benchmark("760M", 512, 32, 10, num_decoder_layer=24, num_header=16, size_per_header=96, max_batch_size=64, vocab_size=51200)
    # b.start()
    # b.to_csv()

    # b = Benchmark("1.3B", 512, 32 , 10, 24, 32, 64, 64, 50304) # 1.3B
    # b.start()
    # b.to_csv()

    # b = Benchmark("2.7B", 512, 32, 10, num_decoder_layer=32, num_header=32, size_per_header=80, max_batch_size=64, vocab_size=51200)
    # b.start()
    # b.to_csv()

    tensor_para_size = 1
    while (tensor_para_size <= 8):
        b = Benchmark(f"5.11B_TP_{tensor_para_size}", 512, 32, 10, num_decoder_layer=24, num_header=32, size_per_header=128, max_batch_size=64, vocab_size=51200, tensor_para_size=tensor_para_size)
        b.start()
        b.to_csv()
        tensor_para_size = tensor_para_size * 2

    tensor_para_size = 1
    while (tensor_para_size <= 8):
        b = Benchmark(f"6.7B_TP_{tensor_para_size}", 512, 32, 10, num_decoder_layer=32, num_header=32, size_per_header=128, max_batch_size=64, vocab_size=51200, tensor_para_size=tensor_para_size)
        b.start()
        b.to_csv()
        tensor_para_size = tensor_para_size * 2

    b = Benchmark("13B", 512, 32, 10, num_decoder_layer=40, num_header=40, size_per_header=128, max_batch_size=64, vocab_size=51200)
    b.start()
    b.to_csv()

    b = Benchmark("89B", 512, 32 , 10, 48, 96, 128, 64, 51200)
    b.start()
    b.to_csv()

    b = Benchmark("175B", 512, 32, 10, num_decoder_layer=96, num_header=96, size_per_header=128, max_batch_size=64, vocab_size=51200)
    b.start()
    b.to_csv()

    # # test size_per_header=160
    # #b = Benchmark("272B", 512, 32, 10, num_decoder_layer=96, num_header=96, size_per_header=160, max_batch_size=32, vocab_size=51200)
    # #b.start()
    # #b.to_csv()

    b = Benchmark("310B", 512, 32, 10, num_decoder_layer=96, num_header=128, size_per_header=128, max_batch_size=1, vocab_size=51200)
    b.start()
    b.to_csv()
