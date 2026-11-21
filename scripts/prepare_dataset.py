"""Explicit local-only dataset preparation. Does not train or download anything."""
import argparse
import os
from glaucoma.config import load_config
from glaucoma.data.loading import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.getenv('PROJECT_CONFIG', 'configs/project.yaml'))
    args = parser.parse_args()
    frame = prepare_dataset(load_config(args.config))
    print(frame.groupby(['split', 'label']).size().to_string())
    if frame.patient_id.fillna('').eq('').all():
        print('WARNING: patient IDs unavailable; image-level splitting cannot establish patient independence.')


if __name__ == '__main__':
    main()
