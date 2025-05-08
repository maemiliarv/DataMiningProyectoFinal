import os
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from utils import setup_logger, load_data, save_model, parse_args
from features import create_features


def main(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup logger
    log_file = cfg.get('logging', {}).get('file')
    logger = setup_logger('train', log_file=log_file)

    # Load and process data
    logger.info('Loading data...')
    df = load_data(cfg['data']['input_path'])
    logger.info('Creating features...')
    df = create_features(df)

    # Prepare inputs and target
    features = cfg['data']['features']
    target = cfg['data']['target']
    X = df[features]
    y = df[target]

    # Train-test split
    logger.info('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg['data'].get('test_size', 0.2),
        random_state=cfg['data'].get('random_state', 42)
    )

    # Initialize and train model
    model_params = cfg['model'].get('params', {})
    model = RandomForestRegressor(**model_params)
    logger.info('Training model...')
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f'Test R^2 score: {score:.4f}')

    # Save artefacts
    model_path = cfg['model']['output_path']
    save_model(model, model_path)
    logger.info(f'Model saved to {model_path}')