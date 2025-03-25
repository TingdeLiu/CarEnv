class Metric:
    def report(self) -> float:
        raise NotImplementedError


class EpisodeAveragingMetric(Metric):
    def __init__(self):
        self._sum = 0.
        self._count = 0

    def update(self, value):
        self._sum += value
        self._count += 1

    def report(self) -> float:
        if self._count > 0:
            return self._sum / self._count
        else:
            return 0.0


class CounterMetric(Metric):
    def __init__(self):
        self._value = 0

    def increment(self, step=1):
        self._value += step

    def set(self, value):
        self._value = value

    def report(self) -> float:
        return self._value
