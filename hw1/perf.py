import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pupok
import time
from typing import Callable

def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


def compare(matrix_size: int) -> None:
    vec_a = np.random.rand(matrix_size)
    vec_b = np.random.rand(matrix_size)

    list_a = vec_a.tolist()
    list_b = vec_b.tolist()

    print(
        "Cos sim (C++), size={0}: {1} seconds".format(
            matrix_size, test_timings(pupok.cosine_similarity, list_a, list_b)
        )
    )
    print(
        "Cos sim (Python scipy), size={0}: {1} seconds\n".format(
            matrix_size, test_timings(cosine_similarity, vec_a.reshape(-1, 1), vec_b.reshape(-1, 1))
        )
    )

if __name__ == "__main__":
    for size in [100, 1000, 2000, 3000, 6000, 10000]:
        compare(size)