from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a rolling series and its global average."""

    def __init__(self, window_size=20, fmt=None) -> None:
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        if not is_dist_avail_and_initialized():
            return
        tensor = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(tensor)
        count, total = tensor.tolist()
        self.count = int(count)
        self.total = total

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t") -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (float, int)):
                raise TypeError(f"Metric {key!r} must be numeric, got {type(value).__name__}")
            self.meters[key].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter!s}" for name, meter in self.meters.items())

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter) -> None:
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        total_items = len(iterable)
        space_fmt = ":" + str(len(str(total_items))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ]
        )
        if torch.cuda.is_available():
            log_msg += self.delimiter + "max mem: {memory:.0f}"

        for index, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == total_items - 1:
                eta_seconds = iter_time.global_avg * (total_items - index)
                values = dict(
                    eta=str(datetime.timedelta(seconds=int(eta_seconds))),
                    meters=str(self),
                    time=str(iter_time),
                    data=str(data_time),
                )
                if torch.cuda.is_available():
                    values["memory"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print(log_msg.format(index, total_items, **values))
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        seconds_per_item = total_time / total_items if total_items else 0.0
        print(f"{header} Total time: {total_time_str} ({seconds_per_item:.4f} s / it)")


def count_parameters_in_MB(model) -> float:
    """Return the number of trainable parameters, in millions."""
    return (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6
    )


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    """Return crop coordinates and resize target for the frame loader."""
    if is_train:
        left = np.random.randint(0, resize[0] - crop_size)
        top = np.random.randint(0, resize[1] - crop_size)
    else:
        left = (resize[0] - crop_size) // 2
        top = (resize[1] - crop_size) // 2
    return (left, top, left + crop_size, top + crop_size), resize
