# !/usr/bin/env python3

class LearningRateSchedule():
    ''' Implementation of learning rate decay.
    '''
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))
