from collections import deque
import numpy as np
import socket
import threading
import time
import torch.distributed as dist
import torch
import sys
import functools
import os
def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

print_first_nan = True

print_first_nan = True

def check_nan_tensor(tensor):
    global print_first_nan
    if tensor is not None and torch.isnan(tensor).any():
        if print_first_nan:
            return True
    return False

def check_nan_output(output):
    if isinstance(output, torch.Tensor):
        return check_nan_tensor(output)
    elif isinstance(output, tuple) or isinstance(output, list):
        return any(check_nan_output(x) for x in output)
    elif isinstance(output, dict):
        return any(check_nan_output(x) for x in output.values())
    else:
        return any(check_nan_output(getattr(output, attr_name)) for attr_name in dir(output) if not attr_name.startswith('_'))

def check_nan_hook(top_module_name, module, input, output):
    global print_first_nan
    if output is not None and check_nan_output(output):
        if print_first_nan:
            print(f"NaN detected in module: {top_module_name} -> {module}")
            print_first_nan = False  # 只打印第一个NaN的位置

def register_hooks_to_modules(top_module_name, module):
    for sub_module in module.modules():
        sub_module.register_forward_hook(lambda mod, inp, out: check_nan_hook(top_module_name, mod, inp, out))
            
class Stability(object):
    def __init__(self,arguements) -> None:
        super().__init__()
        self.consecutive_anomalies = 0
        self.loss_median_window = arguements.loss_median_window
        self.stability_protection = arguements.stability_protection
        self.anomaly_times = arguements.anomaly_times
        self.skip_steps = arguements.skip_steps
        self.consecutive_anomalies_steps = arguements.consecutive_anomalies_steps
        self.loss_history = deque(maxlen=self.loss_median_window)
        # self.checkpointing = checkpointing
        self.stability_port = 6789
        self.global_rank = dist.get_rank()

    def send_message_to_server(self, port, message):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect(('localhost', port))
            client_socket.sendall(message.encode())
        except ConnectionRefusedError:
            print_rank_0("Could not connect to server. The message was not sent.",self.global_rank)
        finally:
            client_socket.close()


    def is_loss_anomaly(self, loss: float) -> bool:
        if not self.stability_protection:
            return False
        #Don't skip the anomaly data for the first few steps
        if len(self.loss_history) < 5:
            return False
        loss_history_array = np.array(self.loss_history)
        median_loss = np.median(loss_history_array)
        #Only For Debug
        is_anomaly = loss > self.anomaly_times * median_loss
        # print(f"loss: {loss}, anomaly_threshold: {self.anomaly_times * median_loss}")    

        if is_anomaly:
            print(f'Detected a loss anomaly: current loss {loss} exceeds loss anomaly threshold: {self.anomaly_times * median_loss}. ie. {self.anomaly_times} times the median loss {median_loss}')

        return is_anomaly
    

    def handle_anomalies(self):
        print_rank_0('Detected consecutive loss anomalies.',self.global_rank)
        if self.global_rank == 0:
            threading.Thread(target=functools.partial(self.send_message_to_server, self.stability_port, "001")).start()
        time.sleep(5)
        sys.exit(1)


    def sync_anomaly_detection(self, is_anomaly: bool) -> bool:
        """
        Synchronize the anomaly detection status across all nodes using all_reduce.

        Args:
            is_anomaly (bool): The anomaly detection status from the local node.

        Returns:
            bool: The synchronized anomaly detection status across all nodes.
        """
        anomaly_detected_tensor = torch.tensor(is_anomaly, dtype=torch.bool, device=torch.cuda.current_device())
        dist.all_reduce(anomaly_detected_tensor, op=dist.ReduceOp.MAX)
        return bool(anomaly_detected_tensor.item())
    
    def track_loss_anomaly(self, total_loss, world_loss=None):
        if self.stability_protection:
            local_anomaly = self.is_loss_anomaly(total_loss)
            anomaly_detected = False
            anomaly_detected = self.sync_anomaly_detection(local_anomaly)
            if anomaly_detected:
                self.consecutive_anomalies += 1
                print_rank_0(f"Loss anomaly detected skip step. Loss value: {total_loss:.4f} Consecutive anomalies: {self.consecutive_anomalies}", self.global_rank)
                return False
            else:
                self.consecutive_anomalies = 0
                self.loss_history.append(world_loss if world_loss is not None else total_loss)
                return True
        else:
            return True

    def handle_loss_anomalies(self):
        if self.stability_protection:
            if self.consecutive_anomalies >= self.consecutive_anomalies_steps:
                print_rank_0(f"Consecutive anomalies reached {self.consecutive_anomalies}, handling anomalies...", self.global_rank)
                if dist.get_rank() == 0:
                    self.handle_anomalies() 
