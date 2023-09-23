import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-a', dest='name', type=float, nargs='+')
parse.add_argument('-i', dest='intv', type=int)

opt = parse.parse_args()
print(opt)