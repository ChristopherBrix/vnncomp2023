import argparse
import numpy as np
from torchvision import transforms, datasets
import os
import onnxruntime as rt
import re
from PIL import Image
import pandas as pd

DEFAULT_EPSILON = [1, 3, 5, 10, 15]
DEFAULT_NETWORK = ['onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx',
                   'onnx/3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx',
                   'onnx/3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx']


def get_testing_dataset(img_size):
    ds = pd.read_csv('GTSRB_dataset/Test.csv')
    y_test = ds["ClassId"].values
    imgs = ds["Path"].values

    data = []
    for img in imgs:
        image = Image.open('GTSRB_dataset/' + img)
        image = image.resize(img_size)
        data.append(np.array(image))
    X_test = np.array(data).astype(np.float32)

    return X_test, y_test


def write_vnn_spec(x_ds, y_ds, index, eps, dir_path="./", prefix="spec", data_lb=0, data_ub=1, n_class=10, mean=0.0, std=1.0, negate_spec=False):
    x = x_ds[index]
    y = y_ds[index]
    x = np.array(x)
    x_lb = np.clip(x - eps, data_lb, data_ub)
    x_lb = ((x_lb-mean)/std).reshape(-1)
    x_ub = np.clip(x + eps, data_lb, data_ub)
    x_ub = ((x_ub - mean) / std).reshape(-1)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if np.all(mean == 0.) and np.all(std == 1.):
        spec_name = f"{prefix}_idx_{index}_eps_{eps:.5f}.vnnlib"
    else:
        existing_specs = os.listdir(dir_path)
        competing_norm_ids = [int(re.match(f"{prefix}_idx_{index}_eps_{eps:.5f}_n([0-9]+).vnnlib", spec).group(
            1)) for spec in existing_specs if spec.startswith(f"{prefix}_idx_{index}_eps_{eps:.5f}_n")]
        norm_id = 1 if len(competing_norm_ids) == 0 else max(
            competing_norm_ids)+1
        spec_name = f"{prefix}_idx_{index}_eps_{eps:.5f}_n{norm_id}.vnnlib"

    spec_path = os.path.join(dir_path, spec_name)

    with open(spec_path, "w") as f:
        f.write(f"; Spec for sample id {index} and epsilon {eps:.5f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        if negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
            f.write(f"))\n")
    return spec_name


def get_sample_idx(n, seed=42, n_max=10000):
    np.random.seed(seed)
    assert n <= n_max, f"only {n_max} samples are available"
    idx = list(np.random.choice(n_max, n, replace=False))
    return idx


def get_all_spec(n, seed, x_test, y_test, sess, input_name, img_size, epsilon, negate_spec, dont_extend, network, instances, new_instances, time_out):
    mean = np.array(0.0).reshape((1, -1, 1, 1)).astype(np.float32)
    std = np.array(1.0).reshape((1, -1, 1, 1)).astype(np.float32)

    idxs = get_sample_idx(n, seed, n_max=len(x_test))
    spec_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../vnnlib")

    i = 0
    ii = 1
    n_ok = 0
    with open(instances, "w" if new_instances else "a") as f:
        while i < len(idxs):
            idx = idxs[i]
            i += 1
            x = x_test[idx]
            y = y_test[idx]
            x_new = x[np.newaxis, ...]
            pred_onx = sess.run(None, {input_name: x_new})[0]
            y_pred = np.argmax(pred_onx, axis=-1)
            n_ok += sum(y == y_pred)

            if y == y_pred:
                if epsilon == None:
                    for eps in DEFAULT_EPSILON:
                        spec_i = write_vnn_spec(x_test, y_test, idx, eps, dir_path=spec_path, prefix="model_"+str(img_size),
                                                data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=negate_spec)
                        f.write(
                            f"{''if network is None else os.path.join('onnx/',os.path.basename(network))},{os.path.join('vnnlib/',spec_i)},{time_out:.1f}\n")
                else:
                    spec_i = write_vnn_spec(x_test, y_test, idx, epsilon, dir_path=spec_path, prefix="model"+str(img_size),
                                            data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=negate_spec)
                    f.write(
                        f"{''if network is None else os.path.join('onnx/',os.path.basename(network))},{os.path.join('vnnlib/',spec_i)},{time_out:.1f}\n")
            elif not dont_extend:
                # only sample idxs while there are still new samples to be found
                if len(idxs) < len(x_test):
                    tmp_idx = get_sample_idx(
                        1, seed=seed+ii, n_max=len(x_test))
                    ii += 1
                    while tmp_idx in idxs:
                        tmp_idx = get_sample_idx(
                            1, seed=seed + ii, n_max=len(x_test))
                        ii += 1
                    idxs.append(*tmp_idx)
        print(f"{len(idxs)-n_ok} samples were misclassified{''if dont_extend else ' and replacement samples drawn'}.")


def get_img_size(network):
    pattern = r"\d+_(\d+)_(\d+)"
    match = re.search(pattern, network)

    if match:
        return int(match.group(2))
    else:
        return None


def process_network(network, n, seed, epsilon, negate_spec, dont_extend, instances, new_instances, time_out):
    instances_dir = os.path.dirname(instances)
    if not os.path.isdir(instances_dir):
        os.mkdir(instances_dir)

    sess = rt.InferenceSession(network)
    input_name = sess.get_inputs()[0].name

    img_size = get_img_size(network)

    print(
        f"Generating {n} random specs using seed {seed}. Model: {network}")
    x_test, y_test = get_testing_dataset((img_size, img_size))

    get_all_spec(n, seed, x_test, y_test, sess,
                 input_name, img_size, epsilon, negate_spec, dont_extend, network, instances, new_instances, time_out)


def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', type=float, default=None,
                        help='The epsilon for L_infinity perturbation')
    parser.add_argument('--n', type=int, default=10,
                        help='The number of specs to generate')
    parser.add_argument('seed', type=int, default=42,
                        help='Random seed for idx generation')
    parser.add_argument("--network", type=str, default=None,
                        help="Network to evaluate as .onnx file.")
    parser.add_argument('--negate_spec', action="store_true", default=False,
                        help='Generate spec that is violated for correct certification')
    parser.add_argument('--dont_extend', action="store_true", default=False,
                        help='Do not filter for naturally correctly classified images')
    parser.add_argument("--instances", type=str,
                        default="./instances.csv", help="Path to instances file")
    parser.add_argument("--new_instances", action="store_true",
                        default=False, help="Overwrite old instances.csv")
    parser.add_argument('--time_out', type=float, default=300.0,
                        help='time out')

    args = parser.parse_args()

    if args.network is not None:
        process_network(args.network, args.n, args.seed,
                        args.epsilon, args.negate_spec, args.dont_extend, args.instances, args.new_instances, args.time_out)
    else:
        for network in DEFAULT_NETWORK:
            process_network(network, args.n, args.seed,
                            args.epsilon, args.negate_spec, args.dont_extend, args.instances, args.new_instances, args.time_out)


if __name__ == "__main__":
    main()
