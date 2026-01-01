import numpy as np

class BaseScheduler:
    """
    BaseScheduler class to handle common functionality:
    """
    def get_lr(self, initial_lr, iteration):
        """
        Get the learning rate given the initial learning rate and the current iteration
        
        :param self: Description
        :param initial_lr: Description
        :param iteration: Description
        """
        return NotImplementedError
    
class ExponentialDecay(BaseScheduler):
    """
    Step Decay Scheduler:
    decreas the lr every number of steps
    """
    def __init__(self, decay_rate: float):
        self.decay_rate = decay_rate
    def get_lr(self, initial_lr, iteration):
        return initial_lr * (self.decay_rate ** iteration)
    
class StepDecay(BaseScheduler):
    """
    Step Decay Scheduler:
    decreas the lr every number of steps
    """
    def __init__(self, decay_rate: float, step_size: int):
        self.decay_rate = decay_rate
        self.step_size = step_size
    def get_lr(self, initial_lr, iteration):
        step = iteration // self.step_size
        return initial_lr * (self.decay_rate ** step)
    
class CosineScheduler(BaseScheduler):
    """
    Cosine Scheduler:
    decreas the lr at a cosine wave manner
    """
    def __init__(self, eta_min: float, max_iterations):
        self.eta_min = eta_min
        self.max_iterations = max_iterations
    def get_lr(self, initial_lr, iteration):
        fraction = min(iteration / self.max_iterations, 1.0)
        lr = self.eta_min + (initial_lr - self.eta_min) * (1 + np.cos((np.pi * fraction))) / 2
        return lr
