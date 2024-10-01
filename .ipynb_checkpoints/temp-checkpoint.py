import os

import pysvelte
import numpy as np


# Running pysvelte the first time will re-compile the Svelte UI code, which might take a while
# pysvelte.Hello(name="World")


tokens = ["A", "B", "C", "D", "E", "F", "G", "H"]
attention_pattern = np.tril(np.random.normal(loc=0.3, scale=0.2, size=(8, 8, 3)))
vis = pysvelte.AttentionMulti(tokens=tokens, attention=attention_pattern)

vis.publish(os.getcwd() + os.sep + "hello_world.html")
