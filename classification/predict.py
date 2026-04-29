"""
Predict file type from binary fragments (.bin files).

Supports single file or batch prediction from a folder.

Usage:
  # Single file
  python predict.py path/to/fragment.bin

  # Batch: put .bin files in the predict_input/ folder, then run:
  python predict.py predict_input/

  # Choose model
  python predict.py predict_input/ --model rf
  python predict.py predict_input/ --model cnn
  python predict.py predict_input/ --model xgboost
  python predict.py predict_input/ --model resnet
  python predict.py predict_input/ --model mlp_features
  python predict.py predict_input/ --model ensemble
  python predict.py predict_input/ --model all
"""

import os
import sys
import argparse
import numpy as np
import csv

CHUNK_SIZE = 4096

# Model paths
MODELS = {
    'rf': {
        'name': 'Random Forest',
        'path': 'saved_models/random_forest/rf_model.joblib',
        'type': 'sklearn',
    },
    'xgboost': {
        'name': 'XGBoost',
        'path': 'saved_models/xgboost/xgb_model.joblib',
        'type': 'sklearn',
    },
    'svm': {
        'name': 'SVM',
        'path': 'saved_models/svm/svm_model.joblib',
        'type': 'sklearn',
    },
    'cnn': {
        'name': 'CNN',
        'path': 'saved_models/cnn/cnn_model.pth',
        'type': 'torch',
    },
    'resnet': {
        'name': 'ResNet',
        'path': 'saved_models/resnet/resnet_model.pth',
        'type': 'torch',
    },
    'mlp': {
        'name': 'MLP (raw bytes)',
        'path': 'saved_models/mlp/mlp_model.pth',
        'type': 'torch',
    },
    'mlp_features': {
        'name': 'MLP (features)',
        'path': 'saved_models/mlp_features/mlp_features_model.pth',
        'type': 'torch_features',
    },
    'lenet': {
        'name': 'LeNet',
        'path': 'saved_models/lenet/lenet_model.pth',
        'type': 'torch',
    },
    'lstm': {
        'name': 'LSTM',
        'path': 'saved_models/lstm/lstm_model.pth',
        'type': 'torch',
    },
    'ensemble': {
        'name': 'Ensemble (RF+XGB+ResNet)',
        'path': None,  # Ensemble uses component models
        'type': 'ensemble',
        'components': ['rf', 'xgboost', 'resnet'],
        'weights': {'rf': 0.40, 'xgboost': 0.35, 'resnet': 0.25},
    },
}

# Default input folder
DEFAULT_INPUT_FOLDER = "predict_input"


def load_fragment(path):
    """Read a binary fragment file and return normalized numpy array."""
    with open(path, 'rb') as f:
        data = f.read()

    if len(data) != CHUNK_SIZE:
        if len(data) < CHUNK_SIZE:
            data = data + b'\x00' * (CHUNK_SIZE - len(data))
        else:
            data = data[:CHUNK_SIZE]

    arr = np.array(list(data), dtype=np.float32) / 255.0
    return arr


def predict_sklearn(model_path, fragments):
    """Predict using a sklearn/xgboost model saved with joblib."""
    import joblib
    saved = joblib.load(model_path)
    model = saved['model']
    label_enc = saved['label_encoder']
    use_features = saved.get('use_engineered_features', False)

    X = np.array(fragments)

    # Apply feature engineering if model was trained with it
    if use_features:
        from utils.feature_engineering import extract_features_batch
        X = extract_features_batch(X)

    pred_indices = model.predict(X)
    pred_labels = label_enc.inverse_transform(pred_indices)

    all_probs = None
    if hasattr(model, 'predict_proba'):
        all_probs = model.predict_proba(X)

    results = []
    for i in range(len(fragments)):
        probs = None
        if all_probs is not None:
            probs = sorted(zip(label_enc.classes_, all_probs[i]),
                           key=lambda x: x[1], reverse=True)
        results.append((pred_labels[i], probs))

    return results


def predict_torch(model_path, fragments, model_type):
    """Predict using a PyTorch model (CNN or ResNet)."""
    import torch
    import json

    results_path = f'results/{model_type}_results.json'
    if not os.path.exists(results_path):
        return [(None, None)] * len(fragments)

    with open(results_path) as f:
        res = json.load(f)
    # Support both old format (classes at root) and new format (under dataset_info)
    if 'dataset_info' in res and 'classes' in res['dataset_info']:
        class_names = res['dataset_info']['classes']
    elif 'classes' in res:
        class_names = res['classes']
    else:
        print(f"  ❌ No class names found in {results_path}")
        return [(None, None)] * len(fragments)
    num_classes = len(class_names)

    if model_type == 'cnn':
        from models.cnn.train import FragmentCNN
        model = FragmentCNN(CHUNK_SIZE, num_classes)
    elif model_type == 'resnet':
        from models.resnet.train import ResNet1D
        model = ResNet1D(num_classes)
    elif model_type == 'mlp':
        from models.mlp.train import FragmentMLP
        model = FragmentMLP(CHUNK_SIZE, num_classes)
    elif model_type == 'lenet':
        from models.lenet.train import LeNet1D
        model = LeNet1D(CHUNK_SIZE, num_classes)
    elif model_type == 'lstm':
        from models.lstm.train import FragmentLSTM
        model = FragmentLSTM(CHUNK_SIZE, num_classes)
    else:
        print(f"  ❌ Unknown torch model type: {model_type}")
        return [(None, None)] * len(fragments)

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    X = torch.FloatTensor(np.array(fragments)).unsqueeze(1)

    results = []
    with torch.no_grad():
        outputs = model(X)
        probs_all = torch.softmax(outputs, dim=1).numpy()
        for i in range(len(fragments)):
            pred_idx = np.argmax(probs_all[i])
            pred_label = class_names[pred_idx]
            probs = sorted(zip(class_names, probs_all[i]),
                           key=lambda x: x[1], reverse=True)
            results.append((pred_label, probs))

    return results


def predict_torch_features(model_path, fragments, model_type):
    """Predict using a PyTorch model that takes engineered features."""
    import torch
    import json
    from utils.feature_engineering import extract_features_batch

    results_path = f'results/{model_type}_results.json'
    if not os.path.exists(results_path):
        return [(None, None)] * len(fragments)

    with open(results_path) as f:
        res = json.load(f)
    if 'dataset_info' in res and 'classes' in res['dataset_info']:
        class_names = res['dataset_info']['classes']
    elif 'classes' in res:
        class_names = res['classes']
    else:
        print(f"  ❌ No class names found in {results_path}")
        return [(None, None)] * len(fragments)
    num_classes = len(class_names)

    from models.mlp_features.train import FeatureMLP
    model = FeatureMLP(317, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    # Extract features from normalized byte arrays
    X_features = extract_features_batch(np.array(fragments))
    X = torch.FloatTensor(X_features)

    results = []
    with torch.no_grad():
        outputs = model(X)
        probs_all = torch.softmax(outputs, dim=1).numpy()
        for i in range(len(fragments)):
            pred_idx = np.argmax(probs_all[i])
            pred_label = class_names[pred_idx]
            probs = sorted(zip(class_names, probs_all[i]),
                           key=lambda x: x[1], reverse=True)
            results.append((pred_label, probs))

    return results


def predict_ensemble(fragments, filenames):
    """Predict using weighted ensemble of RF + XGBoost + ResNet."""
    import json

    ensemble_info = MODELS['ensemble']
    components = ensemble_info['components']
    weights = ensemble_info['weights']

    # Check all component models exist
    for comp_key in components:
        comp = MODELS[comp_key]
        if comp['path'] and not os.path.exists(comp['path']):
            print(f"  ❌ Ensemble component {comp['name']} not found at {comp['path']}")
            return None

    # Get predictions from each component
    component_results = {}
    for comp_key in components:
        comp = MODELS[comp_key]
        if comp['type'] == 'sklearn':
            results = predict_sklearn(comp['path'], fragments)
        else:
            results = predict_torch(comp['path'], fragments, comp_key)
        component_results[comp_key] = results

    # Combine probabilities
    # Get class names from first component with probabilities
    class_names = None
    for comp_key, results in component_results.items():
        if results and results[0][1]:
            class_names = [label for label, _ in results[0][1]]
            break

    if class_names is None:
        print("  ❌ Could not get class names from component models")
        return None

    ensemble_results = []
    for i in range(len(fragments)):
        # Build probability arrays for each component
        combined_probs = np.zeros(len(class_names))
        for comp_key in components:
            comp_results = component_results[comp_key]
            if comp_results[i][1]:  # has probability info
                prob_dict = {label: prob for label, prob in comp_results[i][1]}
                for j, cls in enumerate(class_names):
                    combined_probs[j] += weights[comp_key] * prob_dict.get(cls, 0.0)

        pred_idx = np.argmax(combined_probs)
        pred_label = class_names[pred_idx]
        probs = sorted(zip(class_names, combined_probs),
                       key=lambda x: x[1], reverse=True)
        ensemble_results.append((pred_label, probs))

    return ensemble_results


def run_batch_prediction(model_key, fragments, filenames):
    """Run prediction for a batch of fragments with a specific model."""
    info = MODELS[model_key]

    if info['type'] == 'ensemble':
        return predict_ensemble(fragments, filenames)

    if info['path'] and not os.path.exists(info['path']):
        print(f"  ❌ {info['name']}: model not found at {info['path']}")
        print(f"     Train it first with: python models/{model_key.replace('rf','random_forest')}/train.py")
        return None

    if info['type'] == 'sklearn':
        return predict_sklearn(info['path'], fragments)
    elif info['type'] == 'torch_features':
        return predict_torch_features(info['path'], fragments, model_key)
    else:
        return predict_torch(info['path'], fragments, model_key)


def print_results(filenames, results, model_name, top_n=3):
    """Print prediction results as a formatted table."""
    print(f"\n{'='*70}")
    print(f"🤖 {model_name} Predictions")
    print(f"{'='*70}")
    print(f"{'File':<40s} {'Prediction':<12s} {'Confidence':<10s}")
    print(f"{'-'*40} {'-'*12} {'-'*10}")

    for fname, (pred, probs) in zip(filenames, results):
        if pred is None:
            print(f"{fname:<40s} {'ERROR':<12s}")
            continue
        conf = f"{probs[0][1]:.1%}" if probs else "N/A"
        print(f"{fname:<40s} .{pred:<11s} {conf:<10s}")

    print()


def save_results_csv(filenames, results, model_name, output_path):
    """Save predictions to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_type', 'confidence', 'model'])
        for fname, (pred, probs) in zip(filenames, results):
            if pred is None:
                continue
            conf = f"{probs[0][1]:.4f}" if probs else "N/A"
            writer.writerow([fname, pred, conf, model_name])


def main():
    parser = argparse.ArgumentParser(
        description="Predict file type from binary fragments"
    )
    parser.add_argument('input', nargs='?', default=DEFAULT_INPUT_FOLDER,
                        help=f'Path to a .bin file or folder of .bin files (default: {DEFAULT_INPUT_FOLDER}/)')
    parser.add_argument('--model', default='rf',
                        choices=['rf', 'xgboost', 'svm', 'cnn', 'resnet', 'mlp', 'mlp_features', 'lenet', 'lstm', 'ensemble', 'all'],
                        help='Model to use for prediction (default: rf)')
    parser.add_argument('--top', type=int, default=3,
                        help='Number of top predictions to show (default: 3)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save predictions to predictions_output.csv')

    args = parser.parse_args()

    # Create default input folder if it doesn't exist
    if args.input == DEFAULT_INPUT_FOLDER and not os.path.exists(DEFAULT_INPUT_FOLDER):
        os.makedirs(DEFAULT_INPUT_FOLDER, exist_ok=True)
        print(f"📁 Created folder: {DEFAULT_INPUT_FOLDER}/")
        print(f"   Place your .bin files there and run this script again.")
        sys.exit(0)

    # Collect files
    if os.path.isdir(args.input):
        files = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.endswith('.bin') and not f.startswith('.')
        ])
        if not files:
            print(f"❌ No .bin files found in {args.input}/")
            print(f"   Place your binary fragment files there and try again.")
            sys.exit(1)
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        print(f"❌ Not found: {args.input}")
        sys.exit(1)

    # Load all fragments
    filenames = [os.path.basename(f) for f in files]
    fragments = [load_fragment(f) for f in files]

    print(f"📁 Loaded {len(fragments)} fragment(s) from: {args.input}")

    # Run predictions
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]

    for model_key in models_to_run:
        info = MODELS[model_key]
        results = run_batch_prediction(model_key, fragments, filenames)

        if results is None:
            continue

        print_results(filenames, results, info['name'], args.top)

        # Show detailed probabilities for single file
        if len(files) == 1:
            pred, probs = results[0]
            if probs:
                print(f"   Top {args.top} predictions:")
                for label, prob in probs[:args.top]:
                    bar = '█' * int(prob * 30)
                    print(f"   {label:>12s}  {prob:6.2%}  {bar}")
                print()

        # Save CSV if requested
        if args.save_csv:
            csv_path = f"predictions_{model_key}.csv"
            save_results_csv(filenames, results, info['name'], csv_path)
            print(f"💾 Saved to: {csv_path}")


if __name__ == "__main__":
    main()
