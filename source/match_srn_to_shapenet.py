import os
from glob import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--srn-path', dest='srn_path', metavar='DATA', help='srn path')
    parser.add_argument('--shapenet-path', dest='shapenet_path', type=str, metavar='PATH', 
                            help='shapenetcore path')

    args = parser.parse_args()

    mapping = {}
    
    category_dirs = glob(os.path.join(args.shapenet_path, "*/"))
    instance_dirs = glob(os.path.join(args.srn_path, "*/"))
    instance_ids = set([os.path.relpath(x, start=args.srn_path) for x in instance_dirs])

    for category_dir in category_dirs:
        category = os.path.relpath(category_dir, start=args.shapenet_path)

        ids_in_category = glob(os.path.join(category_dir, "*/"))
        ids_in_category = set([os.path.relpath(x, start=category_dir) for x in ids_in_category])

        intersection = instance_ids & ids_in_category

        for i in intersection:
            mapping[i] = category
        
        instance_ids = instance_ids - intersection


    print(mapping)

if __name__ == "__main__":
    main()



