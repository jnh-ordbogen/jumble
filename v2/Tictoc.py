import time

class Tictoc:
    def tictoc(func):
        def wrapper(*args):
            t1 = time.time()
            result = func(*args)
            t2 = time.time()-t1

            print(f'Took {(t2)*1000} ms for {func.__name__}')
            return result
        return wrapper
