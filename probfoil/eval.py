from __future__ import print_function


def main(argv=sys.argv[1:]):
    args = argparser().parse_args(argv)

def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('data')
    return parser

if __name__ == '__main__':
    main()
