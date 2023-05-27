import torch
import os

from app.profiler.classifiers.SRAClassifier import SRAClassifier
from dataloaders.SRADataloader import SRADataloader


if __name__ == '__main__':
    experiments: str = 'as'  # string of letters from [sra]

    # optimisation
    device = 'cpu'  # 'gpu' or 'cpu'
    no_workers = min(4, os.cpu_count()//2)
    # using only 50% of capabilities
    if device == 'gpu':
        torch.cuda.set_per_process_memory_fraction(0.5, 0)

    data_filename = 'data_full.csv'
    load = [os.path.join('bests', f'{exp}.ckpt') for exp in experiments]

    if len(set(experiments).difference({'s', 'r', 'a'})) == 0:
        data_dir = 'data'
        model = SRAClassifier(load, no_workers)
        dl = {
            's': iter(SRADataloader('s', data_dir=data_dir, no_workers=no_workers).test_dataloader()),
            'r': iter(SRADataloader('r', data_dir=data_dir, no_workers=no_workers).test_dataloader()),
            'a': iter(SRADataloader('a', data_dir=data_dir, no_workers=no_workers).test_dataloader()),
        }
    else:
        raise Exception(f'Invalid experiment: {experiments}')

    with torch.no_grad():
        correct, full, i = 0, 0, 0

        for s, r, a in zip(dl['s'], dl['r'], dl['a']):
            i += 1
            x, label_s = s
            _, label_r = r
            _, label_a = a

            mapper = {
                's': label_s,
                'r': label_r,
                'a': label_a
            }

            preds = model.predict(x)
            counter = set()
            for k, pred in preds.items():
                t = torch.sub(mapper.get(k), pred)
                t = torch.argwhere(t != 0)
                t = torch.flatten(t)
                counter = counter.union(t.tolist())

            correct += (len(x) - len(counter))
            full += len(x)
            print('Batch:  ', i)

        print('Correct:  ', correct)
        print('Full:     ', full)
        print('Accuracy: ', correct / full)
