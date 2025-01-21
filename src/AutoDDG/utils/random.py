import random

import randomname


def generate_random_name():
    name = randomname.get_name()
    random_number = random.randint(0, 100)
    return f"{name}-{random_number}"
