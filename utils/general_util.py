import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import multiprocessing.pool as pool
import mlflow
from optuna.integration.mlflow import MLflowCallback


def loss_plot(train_loss, valid_loss, file):
    # visualize the loss as the network trained
    plt.ioff() #Turn interactive plotting off
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(file, bbox_inches='tight')
    plt.close(fig)

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TextLogger(object):
    """Writes stream output to external text file.
    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
    """
    A useful logger that
    - writes outputs to files and displays them on the console at the same time.
    - manages the directory of checkpoints and debugging images.
    Args:
        root (str): the root directory of logger
        phase (str): the phase of training.
    """

    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.visualize_directory = os.path.join(self.root, "visualize")
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.visualize_directory, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        # redirect std out
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger
        if phase != 'train':
            self.set_epoch(phase)

    def set_epoch(self, epoch):
        """Set the epoch number. Please use it during training."""
        os.makedirs(os.path.join(self.visualize_directory, str(epoch)), exist_ok=True)
        self.epoch = epoch

    def _get_phase_or_epoch(self):
        if self.phase == 'train':
            return str(self.epoch)
        else:
            return self.phase

    def get_image_path(self, filename: str):
        """
        Get the full image path for a specific filename
        """
        return os.path.join(self.visualize_directory, filename)

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.
        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, when the phase is ``train``, checkpoint will be saved to ``{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.
        """
        if name is None:
            name = self._get_phase_or_epoch()
        name = str(name)
        return os.path.join(self.checkpoint_directory, name)

    def close(self):
        self.logger.close()

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)

class Pool(pool.Pool):
    Process = NoDaemonProcess

def make_mlflow_callback(dir):

    mlrun_dir = f"{dir}/mlruns"
    cb = MLflowCallback(
         tracking_uri=mlrun_dir,
         metric_name="MSE"
     )
    mlflow.set_tracking_uri(mlrun_dir)

    return cb