 
import random
import string

def random_string(N=5):
    chars = string.ascii_letters + string.digits + string.punctuation
    s = ''.join(random.choices(chars, k=N))
    return s