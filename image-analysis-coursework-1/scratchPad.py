import pickle
import numpy as np
a = np.array([1,3,2])# some NumPy array
serialized = pickle.dumps(a, protocol=5) # protocol 0 is printable ASCII
deserialized_a = pickle.loads(serialized)
print(a)