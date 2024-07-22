#!/usr/bin/env python3

from dataclasses import dataclass
import math

@dataclass
class Option:
    name: str
    initial: float
    value: float
    delta: float
    lr: float
    m: float
    v: float

def step(option, args, t):
    m_hat = option.m / (1.0 - args.beta1 ** t)
    v_hat = option.v / (1.0 - args.beta2 ** t)
    option.value -= args.alpha * m_hat / (math.sqrt(option.v) + args.epsilon)

def mv(option, grad, args):
    option.m = args.beta1 * option.m + (1.0 - args.beta1) * grad
    option.v = args.beta2 * option.v + (1.0 - args.beta2) * grad * grad
