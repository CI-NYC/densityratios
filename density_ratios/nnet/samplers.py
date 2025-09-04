from collections.abc import Iterator

import torch
from torch import Generator, Tensor
from torch.utils.data import RandomSampler, TensorDataset


class StablilizedWeightDataset(TensorDataset):
    """Dataset for Stabilized weight estimation.

    For indexes less than the length of the input data (n):
    Each sample will be retrieved by indexing x and y along the first dimension.
    For indexes greater than n:
    Each sample will be obtained by first indexing x and then y separately.
    """

    tensors: tuple[Tensor, ...]

    def __init__(self, x: Tensor, y: Tensor) -> None:
        tensors = [x, y]
        super().__init__(*tensors)
        self.n = self.tensors[0].size(0)

    def __getitem__(self, index):
        d, x_index, y_index = _index_fn(index, self.n)

        x = self.tensors[0][x_index]
        y = self.tensors[1][y_index]
        return (
            torch.hstack((x, y)),
            torch.as_tensor(d, dtype=torch.float32),
            torch.as_tensor(1 / 2, dtype=torch.float32),
        )


def _index_fn(index: int, n: int) -> tuple[bool, int, int]:
    """Index generator

    Example output for n=2 and index between 0 and n*(n+1)
    0 -> (False, 0, 0)
    1 -> (False, 1, 1)
    2 -> (True, 0, 0)
    3 -> (True, 0, 1)
    4 -> (True, 1, 0)
    5 -> (True, 1, 1)
    """
    if index < n:
        x_index, y_index = index, index
        d = False
    else:
        x_index = (index // n) - 1
        y_index = index % n
        d = True

    return d, x_index, y_index


class StablilizedWeightSampler(RandomSampler):
    """Dataset for Stabilized weight estimation.

    Let n=num_samples.
    With probability 1/2 draw a value from 0 to n-1.
    With probability 1/2 draw a value from n to n(n+1) -1.

    When combined with `StablilizedWeightDataset`, this means numerator and
    denominator samples with equal probability.

    Samples from n to n(n+1) -1 can be thought of as an x sample (value mod n),
    and a y sample (value floor divided by n then - 1).

    When replacement is set to true, every set of 2n samples will contain 0 to n-1,
    and n unique samples from from n to n(n+1) -1. The latter samples contain
    0 to n-1 in the x samples and 0 to n-1 in the y samples.

    A samples >=n are said to be deranged when the x sample is different
    from the y sample. See https://en.wikipedia.org/wiki/Derangement.

    This is similar to e.g. Algorithm 1 from Belghazi et al. 2018.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with pseudo-replacement (see above) if ``True``, default=``False``.
        derangement (bool): samples >= n are deranged (see above) if ``True``, default=``False``.
        num_samples (int): number of samples to draw, default=`2 * len(dataset)`.
        generator (Generator): Generator used in sampling.

    References
    ----------
    Belghazi et al. 2018, Mutual Information Neural Estimation, PMLR
    """

    data_source: StablilizedWeightDataset
    replacement: bool
    derangement: bool

    def __init__(
        self,
        data_source,
        replacement: bool = False,
        derangement: bool = False,
        num_samples: int | None = None,
        generator: Generator = None,
    ):
        super().__init__(data_source, replacement, num_samples, generator)
        self.derangement = derangement

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return 2 * len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        # Default Generator set up as in parent class
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            # Sample in batches of 32 like parent class.
            for _ in range(self.num_samples // 32):
                values = _sample_without_replacement(n, 32, generator, self.derangement)
                yield from values.tolist()

            size = self.num_samples % 32
            if size > 0:
                values = _sample_without_replacement(
                    n, size, generator, self.derangement
                )
                yield from values.tolist()

        else:
            for _ in range(self.num_samples // (2 * n)):
                values = _sample_permutation(n, generator, self.derangement)
                yield from values.tolist()

            remaining_samples = self.num_samples % (2 * n)
            if remaining_samples > 0:
                values = _sample_permutation(n, generator, self.derangement)
                yield from values.tolist()[:remaining_samples]


def _sample_permutation(
    n: int, generator: Generator, derangement: bool = False
) -> Tensor:
    x_values = torch.randperm(n, generator=generator)
    y_values = torch.randperm(n, generator=generator)

    if derangement:
        # Derangment probability tends to exp(-1) = 0.367 for large n
        # https://en.wikipedia.org/wiki/Derangement#Growth_of_number_of_derangements_as_n_approaches_%E2%88%9E
        # Use rejection sampling to obtain a derangment
        while torch.any(y_values == x_values):
            x_values = torch.randperm(n, generator=generator)

    disjoint_values = n * (x_values + 1) + y_values
    joint_values = torch.randperm(n, generator=generator)

    return torch.hstack((joint_values, disjoint_values))[torch.randperm(2 * n)]


def _sample_without_replacement(
    n: int, size: int, generator: Generator, derangement: bool = False
) -> Tensor:
    bools = torch.randint(high=2, size=(size,), dtype=torch.bool, generator=generator)
    joint_values = torch.randint(
        high=n, size=(size,), dtype=torch.int64, generator=generator
    )
    x_values = torch.randint(
        high=n, size=(size,), dtype=torch.int64, generator=generator
    )
    y_values = torch.randint(
        high=n, size=(size,), dtype=torch.int64, generator=generator
    )
    if derangement:
        while torch.any(y_values[bools] == x_values[bools]):
            x_values = torch.randint(
                high=n, size=(size,), dtype=torch.int64, generator=generator
            )

    disjoint_values = n * (x_values + 1) + y_values
    return torch.where(bools, disjoint_values, joint_values)
